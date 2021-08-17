#https://pythonacademy.com.br/blog/estruturas-de-repeticao#loops-utilizando-for

lista = [1, 2, 3, 4, 5]

for item in lista:
    print(item)



for item in sequencia:
    print(item)
else:
    print('Todos os items foram exibidos com sucesso')


computador = ['Processador', 'Teclado', 'Mouse']
for item in computador:
    print(item)

notas = {
    'Potuguês': 7, 
    'Matemática': 9, 
    'Lógica': 7, 
    'Algoritmo': 7
}

for chave, valor in notas.items():
    print(f"{chave}: {valor}")


for caractere in 'Python':
    print(caractere)


contador = 0

while contador < 10:
    print(f'Valor do contador é {contador}')
    contador += 1




contador = 0

while contador < 10:
    contador += 1
    print(f'Valor do contador é {contador}')    
else:
    print(f'Fim do while e o valor do contador é {contador}')


for num in range(10):
    # Se o número for igual a = 5, devemos parar o loop
    if num == 5:
        # Break faz o loop finalizar
        break
    else:
        print(num)
        


num = 0
while num < 5:
    num += 1

    if num == 3:
        break
        
    print(num)



for num in range(5):
    if num == 3:
        print("Encontrei o 3")
        # Executa o continue, pulando para o próximo laço
        continue
    else:
        print(num)

    print("Estou abaixo do IF")




num = 0
while num < 5:
    num += 1

    if num == 3:
        continue
        
    print(num)





for item in range(5000):
    pass

while False:
    pass

class Classe:
    pass

if True:
    pass
else:
    pass

def funcao():
    pass



class Classe:

def funcao():
    pass





contador = 0
computador = ['Processador', 'Teclado', 'Mouse']

for elemento in computador:
    print(f"Índice={contador} | Valor={elemento}")
    contador += 1




computador = ['Processador', 'Teclado', 'Mouse']
for indice, valor in enumerate(computador):
    print(f"Índice={indice} | Valor={valor}")




computador = ['Processador', 'Teclado', 'Mouse']
for indice in range(len(computador)):
    print(f"Índice={indice} | valor={computador[indice]}")
