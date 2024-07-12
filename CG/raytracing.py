import numpy as np
import matplotlib.pyplot as plt

def refletido(vetor, eixo):
    return vetor - 2 * np.dot(vetor, eixo) * eixo

def normaliza(vetor):
    return vetor / np.linalg.norm(vetor)

def intersec_esfera(centro, raio, origem_raio, direcao_raio):
    b = 2 * np.dot(direcao_raio, origem_raio - centro)
    c = np.linalg.norm(origem_raio - centro) ** 2 - raio ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def objeto_proximo(objetos, origem_raio, direcao_raio):
    distancias = [intersec_esfera(obj['centro'], obj['raio'], origem_raio, direcao_raio) for obj in objetos]
    objeto_mais_proximo = None
    menor_distancia = np.inf
    for indice, distancia in enumerate(distancias):
        if distancia and distancia < menor_distancia:
            menor_distancia = distancia
            objeto_mais_proximo = objetos[indice]
    return objeto_mais_proximo, menor_distancia

largura = 400
altura = 300

profundidade_maxima = 3

camera = np.array([0, 0, 1])
razao = float(largura) / altura
tela = (-1, 1 / razao, 1, -1 / razao)
luz = { 'posicao': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

objetos = [
    { 'centro': np.array([-0.5, 0, -1.5]), 'raio': 0.4, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'brilho': 50, 'reflexao': 0.5 },
    { 'centro': np.array([0.4, -0.3, -1]), 'raio': 0.2, 'ambient': np.array([0, 0.1, 0.1]), 'diffuse': np.array([0, 0.7, 0.7]), 'specular': np.array([1, 1, 1]), 'brilho': 100, 'reflexao': 0.4 },
    { 'centro': np.array([-0.1, 0.2, -0.5]), 'raio': 0.1, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'brilho': 100, 'reflexao': 0.3 },
    { 'centro': np.array([0.3, 0, -0.8]), 'raio': 0.3, 'ambient': np.array([0.1, 0.1, 0]), 'diffuse': np.array([0.6, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'brilho': 150, 'reflexao': 0.6 },
    { 'centro': np.array([0, -9000, 0]), 'raio': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'brilho': 100, 'reflexao': 0.3 }
]

imagem = np.zeros((altura, largura, 3))
for i, y in enumerate(np.linspace(tela[1], tela[3], altura)):
    for j, x in enumerate(np.linspace(tela[0], tela[2], largura)):
        pixel = np.array([x, y, 0])
        origem = camera
        direcao = normaliza(pixel - origem)

        cor = np.zeros((3))
        reflexao = 1

        for k in range(profundidade_maxima):
            objeto_mais_proximo, menor_distancia = objeto_proximo(objetos, origem, direcao)
            if objeto_mais_proximo is None:
                break

            intersecao = origem + menor_distancia * direcao
            normal_superficie = normaliza(intersecao - objeto_mais_proximo['centro'])
            ponto_deslocado = intersecao + 1e-5 * normal_superficie
            intersecao_luz = normaliza(luz['posicao'] - ponto_deslocado)

            _, menor_distancia = objeto_proximo(objetos, ponto_deslocado, intersecao_luz)
            distancia_intersecao_luz = np.linalg.norm(luz['posicao'] - intersecao)
            sombreado = menor_distancia < distancia_intersecao_luz

            if sombreado:
                break

            iluminacao = np.zeros((3))

            iluminacao += objeto_mais_proximo['ambient'] * luz['ambient']
            iluminacao += objeto_mais_proximo['diffuse'] * luz['diffuse'] * np.dot(intersecao_luz, normal_superficie)
            intersecao_camera = normaliza(camera - intersecao)
            H = normaliza(intersecao_luz + intersecao_camera)
            iluminacao += objeto_mais_proximo['specular'] * luz['specular'] * np.dot(normal_superficie, H) ** (objeto_mais_proximo['brilho'] / 4)

            cor += reflexao * iluminacao
            reflexao *= objeto_mais_proximo['reflexao']

            origem = ponto_deslocado
            direcao = refletido(direcao, normal_superficie)

        imagem[i, j] = np.clip(cor, 0, 1)
    print("%d/%d" % (i + 1, altura))

plt.imsave('cena.png', imagem)
