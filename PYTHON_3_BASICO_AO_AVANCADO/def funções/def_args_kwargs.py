"""
1 - Crie uma função1 que recebe uma função2 como parâmetro e retorne o valor da
função2 executada.
"""

# def ola_mundo():
#     return 'Olá mundo!'
#
#
# def mestre(funcao):
#     return funcao()
#
#
# print(ola_mundo())

# minha versão
 def ola_mundo():
     return 'Olá mundo!'


 def funcao1(funcao2):
     return funcao2()


 print(ola_mundo())

"""
2 - Crie uma função1 que recebe uma função2 como parâmetro e retorne o valor da
função2 executada. Faça a função1 executar duas funções que recebam um número 
diferente de argumentos.
"""

#primeiro faco uma def que recebe nomefunc, args e kwargs
# e que return nomefunc(args kwargs)
def mestre(funcao, *args, **kwargs):
    return funcao(*args, **kwargs)

#depois crio uma def com nomefunc(arg)
#   e que return um f'{}
def fala_oi(nome):
    return f'Oi {nome}'

#por ultimo faco outra que recebe dois args e retorna ambos
def saudacao(nome, saudacao):
    return f'{saudacao} {nome}'


executando = mestre(fala_oi, 'Luiz')
executando2 = mestre(saudacao, 'Luiz', saudacao='Bom dia!')
print(executando)
print(executando2)
