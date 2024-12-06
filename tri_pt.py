import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
import matplotlib.pyplot as plt
#import os

#modelo logistico de tres parametros ML3
def P(theta,alpha,beta,r):
    #chance de acertar
    return r + (1.0-r)*1.0/(1+np.exp(-alpha*(theta-beta)))

def Q(theta,alpha=1,beta=5,r=0.2):
    #chance de errar
    return 1- P(theta,alpha,beta,r)

def Pp(theta,alpha=1,beta=5,r=0.2):
    return 1.0/(1+np.exp(-alpha*(theta-beta)))

def Qp(theta,alpha=1,beta=5,r=0.2):
    return 1- Pp(theta,alpha,beta,r)

def W(theta,alpha=1,beta=5,r=0.2):
    #funcao peso auxiliar
    return Pp(theta,alpha,beta,r)*Qp(theta,alpha,beta,r)/(P(theta,alpha,beta,r)*Q(theta,alpha,beta,r))


def I(theta,r=0.2,alpha=0.1,beta=50):
    p=P(theta,alpha,beta,r)
    return alpha**2*(1-p)*((p-r)/(1-r))**2/p

##################################      CLASSES         ############################################ 
#classe questao
class questao:    
    def __init__(self,N,acertos):
        self.N=N
        self.J=acertos.shape[0]
        self.respostas=acertos[N]
        self.acertos = self.calcacertos(N) 
        self.pAcerto= self.calcPAcerto()
        self.alpha= 1.0
        self.beta=0
        self.r= 0.2
        self.padraoAcertoExp=[]
        self.habilidade=self.calcHabilidade(acertos['Pontos padronizados'])
        
        
    def calcHabilidade(self,pontos):
        pontos_media=pontos.mean()
        pontos_std=pontos.std()
        habilidade=(pontos-pontos_media)/pontos_std
        return habilidade
        
    def estimateParams(self,acertos,nSteps=100):        
        step=int(self.J/nSteps)
        k=0
        pontos_media=acertos['Pontos padronizados'].mean()
        pontos_std=acertos['Pontos padronizados'].std()
        habilidadeSummary=[]
        acertoSummary=[]
        habilidade=acertos.sort_values('Pontos padronizados',ascending=True)['Pontos padronizados']

        habilidade=(habilidade-pontos_media)/pontos_std
        padrao=acertos.sort_values('Pontos padronizados',ascending=True)[self.N]
        while (k+1)*step<self.J:
            habilidadeSummary.append(habilidade[k*step:(k+1)*step].mean())
            acertoSummary.append(padrao[k*step:(k+1)*step].mean())
            k+=1
        popt, pcov = opt.curve_fit(P, habilidadeSummary, acertoSummary,p0=[0.1,3,0.2] ,bounds=([0,-5,0], [1.5,5,1]))
        self.padraoAcertoExp=[np.asarray(habilidadeSummary), np.asarray(acertoSummary) ]
        self.setAlpha(popt[0])
        self.setBeta(popt[1])
        self.setR(popt[2])
        return popt


    def estimateBeta(self):
        fAcerto=self.calcPAcerto()
        def func(x):
            return norm.cdf(1-x)-fAcerto

        return opt.root(func,x0=0).x[0]

    #u=1 acertou, u=0 errou.
    def calcacertos(self,N):
        return self.respostas

    #funcoes de calculo de parametros 
    def calcParams(self):      
        def custo (xi):
            self.setAlpha(xi[0])
            self.setBeta(xi[1])
            self.setR(xi[2])
            ca,cb,cr=0,0,0
            for j in range(self.J):
                ca+=(1.0-self.r) *( self.acertos[j] - self.P(self.habilidade[j]) )*self.W(self.habilidade[j])*(self.habilidade[j] - self.beta)
                cb+=0.0 - self.alpha*(1.0-self.r) *( self.acertos[j] - self.P(self.habilidade[j]) )*self.W(self.habilidade[j])
                cr+=( self.acertos[j] - self.P(self.habilidade[j]) )*self.W(self.habilidade[j])/self.Pp(self.habilidade[j])            
            return [ca,cb,cr]
        
        paramOpt=opt.root(custo,x0=[self.alpha,self.beta,self.r])
        
        newAlpha=paramOpt.x[0]
        newBeta=paramOpt.x[1]
        newR=paramOpt.x[2]
        self.setAlpha(newAlpha)
        self.setBeta(newBeta)
        self.setR(newR)
        return paramOpt.x          
        
    
    def calcAlpha(self):
        #esse custo deve ser zerado para determinar o parâmetro alpha
        def custoAlpha(alpha):
            self.setAlpha(alpha)
            i=1
            custo=0
            for j in range(self.J):
                #print (j)
                custo+= (1.0-self.r) *( self.acertos[j] - self.P(self.habilidade[j]) )*self.W(self.habilidade[j])*(self.habilidade[j] - self.beta)
            return custo
        
        alfaOpt=opt.root(custoAlpha,x0=self.alpha)
        newAlpha=alfaOpt.x[0]
        self.setAlpha(newAlpha)
        return newAlpha  
    
    def calcBeta(self):
        def custoBeta(beta):
            self.setBeta(beta)
            custo=0
            for j in range(self.J):
                #print (j)
                custo+= 0.0 - self.alpha*(1.0-self.r) *( self.acertos[j] - self.P(self.habilidade[j]) )*self.W(self.habilidade[j])
            return custo
        #print(custoBeta(0.43),custoBeta(0.5))
        betaOpt=opt.root(custoBeta,x0=self.beta)
        newBeta=betaOpt.x[0]
        self.setBeta(newBeta)
        return newBeta    
    
    def calcR(self):
        def custoR(r):
            self.setR(r)
            custo=0
            for j in range(self.J):
                #print (j)
                custo+= ( self.acertos[j] - self.P(self.habilidade[j]) )*self.W(self.habilidade[j])/self.Pp(self.habilidade[j])
            return custo
        rOpt=opt.root(custoR,x0=self.r)
        newR=rOpt.x[0]
        self.setR(newR)        
        return newR   
    


    def verossemelhanca(self,param): 
        #menos Log_verossemelhança. Ou seja, essa funcao deve ser minimizada para um bom ajuste ML3.
        self.setAlpha(param[0])
        self.setBeta(param[1])
        self.setR(param[2]) 
        soma = self.acertos*np.log( self.P(self.habilidade) ) + (1- self.acertos) * np.log( self.Q(self.habilidade) ) 
        return -soma.sum()
    
    def logL(self):
        soma=0
        for j in range(self.J):    
            soma += self.acertos[j]*np.log( self.P(self.habilidade[j]) ) + (1- self.acertos[j]) * np.log( self.Q(self.habilidade[j]) ) 
        return soma
        
    def jacobian(self,param):
        self.setAlpha(param[0])
        self.setBeta(param[1])
        self.setR(param[2])
        somaA=0
        somaB=0
        somaC=0
        for j in range(self.J): 
            theta=self.habilidade[j]
            fator= (self.acertos[j]-self.P(self.habilidade[j]) )/(self.P(self.habilidade[j]) *self.Q(self.habilidade[j]))
            somaA += -fator * (1-self.r)*(theta-self.beta)*np.exp(-self.alpha*(theta-self.beta))/(1+np.exp(-self.alpha*(theta-self.beta)))**2
            somaB += -fator * (1-self.r)*self.alpha*np.exp(-self.alpha*(theta-self.beta))/(1+np.exp(-self.alpha*(theta-self.beta)))**2
            somaC += fator * np.exp(-self.alpha*(theta-self.beta))/(1+np.exp(-self.alpha*(theta-self.beta)))
        
        return [-somaA,-somaB,-somaC]
    
    def optParameters(self,method='Powell'):
        if method=='Powell':
            x0=[self.alpha,self.beta,self.r]
            res = opt.minimize(self.verossemelhanca, x0, method='Powell',bounds=((0, 1.5), (-5, 5),(0, 1.0)),tol=1e-4)
            #print (res)
            self.setAlpha(res.x[0])
            self.setBeta(res.x[1])
            self.setR(res.x[2])
            return res
        
        if method=='BFGS':
            x0=[self.alpha,self.beta,self.r]        
            
            
            res = opt.minimize(self.verossemelhanca, x0, jac=self.jacobian, method='BFGS',tol=1e-4)
            #print (res)
            self.setAlpha(res.x[0])
            self.setBeta(res.x[1])
            self.setR(res.x[2])
            return res
    
    def calcPAcerto(self):
        return self.acertos.mean()    
    def params(self):
        return self.alpha,self.beta,self.r    
    #funcoes de modificacao de parametros
    def setAlpha(self,alpha):
        self.alpha=alpha
        return alpha
    def setBeta(self,beta):
        self.beta=beta
        return beta
    def setR(self,r):
        self.r=r
        return r    
    
    #calculo de funcoes de probabilidade
    def P(self,theta):
        #modelo logistico de tres parametros ML3
        return self.r + (1.0-self.r)*1.0/(1+np.exp(-self.alpha*(theta-self.beta)))
    
    def Q(self,theta):
        return 1- self.P(theta)
    def Pp(self,theta):
        return 1.0/(1+np.exp(-self.alpha*(theta-self.beta)))
    def Qp(self,theta):
        return 1- self.Pp(theta)
    def W(self,theta):
        #funcao peso auxiliar
        return self.Pp(theta)*self.Qp(theta)/(self.P(theta)*self.Q(theta))
    
    def I(self,theta,r=0.2,alpha=0.1,beta=50):
        p=self.P(theta) 
        return self.alpha**2*(1-p)*((p-self.r)/(1-self.r))**2/p

    def makeFigure(self,targetFolder='./', fileName='question', fileFormat='.eps', refCurve=True,title='Question',meanHab=0, stdHab=1,showExp=True):
        x=np.linspace(0,100,100)
        
        if showExp:
            plt.plot(self.padraoAcertoExp[0]*stdHab+meanHab  ,self.padraoAcertoExp[1],linewidth=0,marker='o' ,alpha=0.1,color='blue',markersize=1.5);
        
        #referenceCurve
        if refCurve:
            plt.plot(x,P(x,r=0.15,alpha=0.1,beta=65),label='Ref', color='red')
            plt.ylim(0,1)
            plt.plot(x,100*I(x,r=0.15,alpha=0.1,beta=65),linestyle='dashed',  color='red')
        
        #questionCurve
        plt.plot(x,self.P((x-meanHab)/stdHab),label=title)
        plt.plot(x,1*self.I((x-meanHab)/stdHab),linestyle='dashed', color=plt.gca().lines[-1].get_color())
        
        

        plt.legend()
        plt.grid()
        plt.savefig(targetFolder+fileName+ fileFormat);
        plt.cla();
        

    
##############   QUESTAO ABERTA    
    
class questaoAberta(questao):  
    def __init__(self,index,acertos):
        self.index=index
        self.acertos=acertos
        self.J=acertos.shape[0]
        self.classificarRespostas()
        self.alpha= 0.15
        #self.betaD=-2
        self.betaC=-2.5
        self.betaB=0
        self.betaA=1.5
        self.habilidade=self.calcHabilidade(acertos['habilidade0'])
        

    def setBetaC(self,beta):
        self.betaC=beta
        return beta
    def setBetaB(self,beta):
        self.betaB=beta
        return beta
    def setBetaA(self,beta):
        self.betaA=beta
        return beta
        
    def classificarRespostas(self):
        questaoIndex=self.index
        corteC=0.09
        corteB=0.49
        corteA=0.91
        self.acertos['isD']=(self.acertos[questaoIndex]<corteC)
        self.acertos['isC']=( (self.acertos[questaoIndex]>corteC) & (self.acertos[questaoIndex] < corteB) )
        self.acertos['isB']=( (self.acertos[questaoIndex]>corteB) & (self.acertos[questaoIndex] < corteA) )
        self.acertos['isA']=(self.acertos[questaoIndex]>corteA )
        self.acertos[['isD','isC','isB', 'isA']] = self.acertos[['isD','isC','isB', 'isA']].replace({True: 1, False: 0})
        return True
        

    def PAberta(self, theta):
        expA=np.exp(-self.alpha*(theta-self.betaA))
        expB=np.exp(-self.alpha*(theta-self.betaB))
        expC=np.exp(-self.alpha*(theta-self.betaC))
        #expD=np.exp(self.alpha*(theta-self.betaD))
        return [expC/(1+expC) , 1.0/(1+expC) - 1.0/(1+expB) , 1.0/(1+expB) - 1.0/(1+expA) , 1.0/(1+expA) ]


    def logL(self):   
        self.acertos['logLD']=(self.acertos['isD'])*np.log( self.PAberta(self.habilidade)[0] ) + (1.0- self.acertos['isD'])*np.log(1- self.PAberta(self.habilidade)[0] )
        self.acertos['logLC']=(self.acertos['isC'])*np.log( self.PAberta(self.habilidade)[1] ) + (1.0- self.acertos['isC'])*np.log(1- self.PAberta(self.habilidade)[1] )
        self.acertos['logLB']=(self.acertos['isB'])*np.log( self.PAberta(self.habilidade)[2] ) + (1.0- self.acertos['isB'])*np.log(1- self.PAberta(self.habilidade)[2] )
        self.acertos['logLA']=(self.acertos['isA'])*np.log( self.PAberta(self.habilidade)[3] ) + (1.0- self.acertos['isA'])*np.log(1- self.PAberta(self.habilidade)[3] )
        #soma += self.acertos[j]*np.log( self.P(self.habilidade[j]) ) + (1- self.acertos[j]) * np.log( self.Q(self.habilidade[j]) ) 
        return self.acertos['logLD'].sum()+self.acertos['logLC'].sum()+self.acertos['logLB'].sum()+self.acertos['logLA'].sum()
    
    def minusVerossemelhanca(self,param): 
        #menos Log_verossemelhança. Ou seja, essa funcao deve ser minimizada para um bom ajuste ML3.
        self.setAlpha(param[0])
        self.setBetaC(param[1])
        self.setBetaB(param[1]+param[2])
        self.setBetaA(param[1]+param[2]+param[3])
        return -1.0*self.logL()


    def optParameters(self):
            x0=[self.alpha,self.betaC,self.betaB-self.betaC,self.betaA-self.betaB]
            res = opt.minimize(self.minusVerossemelhanca, x0, method='Powell',bounds=((0, 1.5), (-5.0, 5.0),(-0.0, 5.0),(-0.0, 5.0)),tol=1e-4)
            #print (res)
            self.setAlpha(res.x[0])
            self.setBetaC(res.x[1])
            self.setBetaB(res.x[1]+res.x[2])
            self.setBetaA(res.x[1]+res.x[2]+res.x[3])
            return res


        
    
##################################################################    
######################        CLASSE ALUNO  ###################### 
##################################################################

class Aluno:
    def __init__(self,aluno_index,acertos,parametros):
        self.index=aluno_index
        self.data=acertos[acertos['candidato']==aluno_index]
        self.parametros=parametros
        self.acertos=self.getAcertos()
        self.habilidade=float(self.data['habilidade'])

        
    def getAcertos(self):
        I=60 #numero de questoes
        acertos=[]
        for k in range(1,61):
            acertos.append(self.data[k])
        return np.asarray(acertos)
        
        
    def logL(self,theta):
        self.parametros['nula']=((self.parametros['tAcerto']>0.999) | ((self.parametros['questao']>=25) & (self.parametros['questao']<=36)) )
        self.parametros['nula'].replace({True: 1.0, False:0.0})
        
        self.parametros['P']=self.parametros['rs'] + (1.0-self.parametros['rs'])*1.0/(1+np.exp(-self.parametros['alphas0']*(theta-self.parametros['betas0'])))
        self.parametros['Q']=1-self.parametros['P']
        self.parametros['acerto_ij']=self.acertos
        self.parametros['logL_ij']=(1-self.parametros['nula'])*((self.parametros['acerto_ij'])*np.log(self.parametros['P'])+(1-self.parametros['acerto_ij'])*np.log(self.parametros['Q']))
        return self.parametros['logL_ij'].sum()
        


    def minuslogL(self,theta):
        return -self.logL(theta)
    
    def setHabilidade(self, theta):
        self.habilidade=float(theta)
        return self.habilidade

    def calcularHabilidade(self,method='NM'):
        I=len(self.parametros)
        theta0=self.habilidade

        if method=='NM': #nelder-mead
            res = opt.minimize(self.minuslogL,  theta0, method='Nelder-Mead',tol=1e-4)
            #print (res)
            theta=res.x[0]
            self.setHabilidade(theta)
            return theta

        if method=='NR':
            for k in range(len(acertos)):
                theta0=acertos['habilidade'][k]
                
    def print(self):
        print('index:',self.index)
        print('habilidade:',self.habilidade)
        print('log(L):',self.logL(self.habilidade))
        return True
    
    
    
##################################      CODIGOS NAO VALIDADOS         ############################################
## METODO DE NEWTON-RAPHSON

def NewtonRaphOpt(questao):
    
        def h_ji (questao,theta):
            return np.array([(1.0-questao.r)*(theta - questao.beta), -questao.alpha*(1-questao.r), 1.0/questao.Pp(theta)] )
        
        def H_ji(questao,theta):        
            H11=(1-questao.r)*(1- 2*questao.Pp(theta) )*(theta - questao.beta)**2
            H12= -(1 - questao.r)*( 1 + questao.alpha*(theta - questao.beta)*(1- 2*questao.Pp(theta) ))
            H13=-(theta - questao.beta)
            H22=(questao.alpha**2)*(1 - questao.r)*(1- 2*questao.Pp(theta))
            H23=questao.alpha
            H_ji= np.matrix([[H11, H12,H13],[H12, H22, H23],[H13, H23, 0]])
            return H_ji
        
        h=np.array([0.0,0.0,0.0])
        for j in range(questao.J):
            h+= (questao.acertos[j]-questao.P(habilidade[j]))*questao.W(habilidade[j])*h_ji(questao,habilidade[j])
        h=np.asmatrix(h)
        
        H=np.zeros((3,3))
        for j in range(questao.J):
            hji=np.asmatrix( h_ji (questao,habilidade[j]))
            #Newton-Raphson
            #H+=(questao.acertos[j]-questao.P(habilidade[j]))*questao.W(habilidade[j])*(H_ji(questao,habilidade[j]) - (questao.acertos[j]-questao.P(habilidade[j]))*questao.W(habilidade[j])* hji.transpose() * hji      )       
            #scorinf-fischer
            H+= -questao.Pp(habilidade[j])*questao.Qp(habilidade[j])*questao.W(habilidade[j])*hji.transpose() * hji   
        
        xi=np.asmatrix(  np.array([questao.alpha,questao.beta,questao.r]))
        
        
        newxi= (xi.transpose() - np.linalg.inv(H) *h.transpose()).transpose()     
        questao.setAlpha(newxi[0,0])
        questao.setBeta(newxi[0,1])
        questao.setR(newxi[0,2])
        return - np.linalg.inv(H) *h.transpose() ,newxi



