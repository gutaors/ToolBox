#import vendas.calc_preco
#from vendas import calc_preco
from vendas.calc_preco import aumento, reducao
from vendas.formata import preco

preco = 49.90
preco_com_aumento = aumento(valor=preco, porcentagem=15, formata=True)
#preco_com_aumento = vendas.calc_preco.aumento(preco, 15)
print(preco_com_aumento)

preco_com_reducao = reducao(valor=preco, porcentagem=15)
print(preco_com_reducao)