def saudacao(msg='olá', nome='usuário'):
    nome = nome.replace('e','3')
    msg = msg.replace('e','3')
    return f'{msg} {nome}'


variavel = saudacao()
print(variavel)

saudacao(nome = 'zezinho', msg='oi')
saudacao('oi', 'luiz')
saudacao('hello', 'maria')
saudacao('olá', 'otavio')
saudacao('olá', 'joão')

def funcao(var):
    print(var)

variavel = funcao('valor que eu quero')

if variavel:
    print(variavel)
else
print('nenhum valor')


def divisao(n1, n2):
    if n2 == 0: 
        return
    
    return n1 / n2

divide = divisao(8, 4)

if divide:
    print(divide)
else
    print("conta inválida")


def dumb():
    return 1.1

var = dumb()
print(var,type(var))
print(dumb(), type(dumb()))
