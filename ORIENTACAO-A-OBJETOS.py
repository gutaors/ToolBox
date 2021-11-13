#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>Data Science Academy - Python Fundamentos - Capítulo 5</font>
# 
# ## Download: http://github.com/dsacademybr

# In[7]:


# Versão da Linguagem Python
from platform import python_version
print('Versão da Linguagem Python Usada Neste Jupyter Notebook:', python_version())


# In[8]:


# CLASSE

# Criando uma classe chamada Livro 
# titulo'O Monge e o Executivo'
# ISBN 9988888
class Livro():
    def __init__(self):
        self.titulo = 'O Monge e o Executivo'
        self.isbn = 9988888
        print("Construtor chamado para criar um objeto desta classe")
    def imprime(self):
        print("Foi criado o livro %s e ISBN %d" %(self.titulo, self.isbn))

#OBJETO
        
# Criando uma classe
class Estudantes:
    def __init__(self, nome, idade, nota):
        self.nome = nome
        self.idade = idade
        self.nota = nota  
# Criando um objeto chamado Estudante1 a partir da classe Estudantes
Estudante1 = Estudantes("Pele", 12, 9.5)

# METODO

# Criando uma classe chamada Circulo
class Circulo():
    
    # O valor de pi é constante
    pi = 3.14

    # Quando um objeto desta classe for criado, este método será executado e o valor default do raio será 5.
    def __init__(self, raio = 5):
        self.raio = raio 

        
# HERANÇA

# Criando a classe Animal - Super-classe
class Animal():
    
    def __init__(self):
        print("Animal criado")

    def Identif(self):
        print("Animal")

    def comer(self):
        print("Comendo")

# Criando a classe Cachorro - Sub-classe
class Cachorro(Animal):
    
    def __init__(self):
        Animal.__init__(self)
        print("Objeto Cachorro criado")

    def Identif(self):
        print("Cachorro")

    def latir(self):
        print("Au Au!")
        
# Criando um objeto (Instanciando a classe)
rex = Cachorro()


# # Classes

# In[9]:


# ESQUELETO DE UMA CLASSE:


class NomeClasse():
    
    # Este método vai inicializar cada objeto criado a partir desta classe
    # O nome deste método é __init__
    # (self) é uma referência a cada atributo de um objeto criado a partir desta classe
    
    #sendo pragmático, a receita é
    #def
    #__init__
    #(self):
    
    def __init__(self, atributo1, atributo2):
        
        # Atributos de cada objeto criado a partir desta classe. 
        # O self indica que estes são atributos dos objetos
        
        # a receita é self.adsadafas = ''
        self.atributo1 = 'Nome por exemplo'
        self.atributo2 = 9988888
        print("Construtor chamado para criar um objeto desta classe")
        
    # Métodos são funções, que recebem como parâmetro atributos do objeto criado   
    # a receita é def nomeMetodo(self):
    #    acaodometodo
    def metodo1(self):
        print("Foi criado o nome %s e CODIGO %d" %(self.atributo1, self.atributo2))

#esqueleto de um objeto
    # Criando um objeto chamado Func1 a partir da classe Funcionarios
NomeObjeto = NomeClasse("Obama", 20000)


# Para criar uma classe, utiliza-se a palavra reservada class. O nome da sua classe segue a mesma convenção de nomes
# para criação de funções e variáveis, mas normalmente se usa a primeira letra maiúscula em cada palavra no nome da 
# classe.

# In[10]:


# Criando uma classe chamada Livro
class Livro():
    
    # Este método vai inicializar cada objeto criado a partir desta classe
    # O nome deste método é __init__
    # (self) é uma referência a cada atributo de um objeto criado a partir desta classe
    def __init__(self):
        
        # Atributos de cada objeto criado a partir desta classe. 
        # O self indica que estes são atributos dos objetos
        self.titulo = 'O Monge e o Executivo'
        self.isbn = 9988888
        print("Construtor chamado para criar um objeto desta classe")
        
    # Métodos são funções, que recebem como parâmetro atributos do objeto criado    
    def imprime(self):
        print("Foi criado o livro %s e ISBN %d" %(self.titulo, self.isbn))


# In[11]:


# Criando uma instância da classe Livro
Livro1 = Livro()


# In[12]:


# Tipo do Objeto Livro1
type(Livro1)


# In[13]:


# Atributo do objeto Livro1
Livro1.titulo


# In[14]:


# Método do objeto Livro1
Livro1.imprime()


# In[15]:


# Criando a classe Livro com parâmetros no método construtor
class Livro():
    def __init__(self, titulo, isbn):
        self.titulo = titulo
        self.isbn = isbn
        print("Construtor chamado para criar um objeto desta classe")
        
    def imprime(self, titulo, isbn):
        print("Este é o livro %s e ISBN %d" %(titulo, isbn))


# In[16]:


# Criando o objeto Livro2 que é uma instância da classe Livro
Livro2 = Livro("A Menina que Roubava Livros", 77886611)


# In[17]:


Livro2.titulo


# In[18]:


# Método do objeto Livro2
Livro2.imprime("A Menina que Roubava Livros", 77886611)


# In[19]:


# Criando a classe cachorro
class Cachorro():
    def __init__(self, raça):
        self.raça = raça
        print("Construtor chamado para criar um objeto desta classe")


# In[20]:


# Criando um objeto a partir da classe cachorro
Rex = Cachorro(raça='Labrador')


# In[21]:


# Criando um objeto a partir da classe cachorro
Golias = Cachorro(raça='Huskie')


# In[22]:


# Atributo da classe cachorro, utilizado pelo objeto criado
Rex.raça


# In[23]:


# Atributo da classe cachorro, utilizado pelo objeto criado
Golias.raça


# # Objetos

# ## Em python tudo é objeto
# 

# In[24]:


# Criando uma lista
lst_num = ["Data", "Science", "Academy", "Nota", 10, 10]


# In[25]:


# A lista lst_num é um objeto, uma instância da classe lista em Python
type(lst_num)


# In[26]:


lst_num.count(10)


# In[27]:


# Usamos a função type, para verificar o tipo de um objeto
print(type(10))
print(type([]))
print(type(()))
print(type({}))
print(type('a'))


# In[28]:


# Criando um novo tipo de objeto chamado Carro
class Carro(object):
    pass

# Instância do Carro
palio = Carro()

print(type(palio))


# In[29]:


# Criando uma classe
class Estudantes:
    def __init__(self, nome, idade, nota):
        self.nome = nome
        self.idade = idade
        self.nota = nota


# In[30]:


# Criando um objeto chamado Estudante1 a partir da classe Estudantes
Estudante1 = Estudantes("Pele", 12, 9.5)


# In[31]:


# Atributo da classe Estudante, utilizado por cada objeto criado a partir desta classe
Estudante1.nome


# In[32]:


# Atributo da classe Estudante, utilizado por cada objeto criado a partir desta classe
Estudante1.idade


# In[33]:


# Atributo da classe Estudante, utilizado por cada objeto criado a partir desta classe
Estudante1.nota


# In[34]:


# Criando uma classe
class Funcionarios:
    def __init__(self, nome, salario):
        self.nome = nome
        self.salario = salario

    def listFunc(self):
        print("O nome do funcionário é " + self.nome + " e o salário é R$" + str(self.salario))


# In[35]:


# Criando um objeto chamado Func1 a partir da classe Funcionarios
Func1 = Funcionarios("Obama", 20000)


# In[36]:


# Usando o método da classe
Func1.listFunc()


# In[37]:


print("**** Usando atributos *****")


# In[38]:


hasattr(Func1, "nome")


# In[39]:


hasattr(Func1, "salario")


# In[40]:


setattr(Func1, "salario", 4500)


# In[41]:


hasattr(Func1, "salario")


# In[42]:


getattr(Func1, "salario")


# In[43]:


delattr(Func1, "salario")


# In[44]:


hasattr(Func1, "salario")


# # Métodos

# In[45]:


# Criando uma classe chamada Circulo
class Circulo():
    
    # O valor de pi é constante
    pi = 3.14

    # Quando um objeto desta classe for criado, este método será executado e o valor default do raio será 5.
    def __init__(self, raio = 5):
        self.raio = raio 

    # Esse método calcula a área. Self utiliza os atributos deste mesmo objeto
    def area(self):
        return (self.raio * self.raio) * Circulo.pi

    # Método para gerar um novo raio
    def setRaio(self, novo_raio):
        self.raio = novo_raio

    # Método para obter o raio do círculo
    def getRaio(self):
        return self.raio


# In[46]:


# Criando o objeto circ. Uma instância da classe Circulo()
circ = Circulo()


# In[47]:


# Executando um método da classe Circulo
circ.getRaio()


# In[48]:


# Criando outro objeto chamado circ1. Uma instância da classe Circulo()
# Agora sobrescrevendo o valor do atributo
circ1 = Circulo(7)


# In[49]:


# Executando um método da classe Circulo
circ1.getRaio()


# In[50]:


# Imprimindo o raio
print ('O raio é: ', circ.getRaio())


# In[51]:


# Imprimindo a area
print('Area igual a: ', circ.area())


# In[52]:


# Gerando um novo valor para o raio do círculo
circ.setRaio(3)


# In[53]:


# Imprimindo o novo raio
print ('Novo raio igual a: ', circ.getRaio())


# # Herança

# In[54]:


# Criando a classe Animal - Super-classe
class Animal():
    
    def __init__(self):
        print("Animal criado")

    def Identif(self):
        print("Animal")

    def comer(self):
        print("Comendo")
        


# In[55]:


# Criando a classe Cachorro - Sub-classe
class Cachorro(Animal):
    
    def __init__(self):
        Animal.__init__(self)
        print("Objeto Cachorro criado")

    def Identif(self):
        print("Cachorro")

    def latir(self):
        print("Au Au!")


# In[56]:


# Criando um objeto (Instanciando a classe)
rex = Cachorro()


# In[57]:


# Executando o método da classe Cachorro (sub-classe)
rex.Identif()


# In[58]:


# Executando o método da classe Animal (super-classe)
rex.comer()


# In[59]:


# Executando o método da classe Cachorro (sub-classe)
rex.latir()


# # Métodos Especiais

# In[60]:


# Criando a classe Livro
class Livro():
    def __init__(self, titulo, autor, paginas):
        print ("Livro criado")
        self.titulo = titulo
        self.autor = autor
        self.paginas = paginas
                
    def __str__(self):
        return "Título: %s , autor: %s, páginas: %s "     %(self.titulo, self.autor, self.paginas)

    def __len__(self):
        return self.paginas
    
    def len(self):
        return print("Páginas do livro com método comum: ", self.paginas)


# In[61]:


livro1 = Livro("Os Lusíadas", "Luis de Camões", 8816)


# In[62]:


# Métodos especiais
print(livro1)


# In[63]:


str(livro1)


# In[64]:


len(livro1)


# In[65]:


livro1.len()


# In[66]:


# Ao executar a função del para remover um atributo, o Python executa:
# livro1.__delattr__("paginas")
del livro1.paginas


# In[67]:


hasattr(livro1, "paginas")


# In[ ]:




