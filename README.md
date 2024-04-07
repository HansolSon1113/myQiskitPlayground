양자컴퓨터를 활용한 베이즈 정리 - 30508 손한솔
(작성된 모든 코드는 http://myalarm.space:8888/ 에서 확인할 수 있다.)(비밀번호: notebook) - 학교 네트워크 안됨
1. 선정 동기: 작년에 조사했던 양자컴퓨터가 머신 러닝 분야에서 유용하게 사용될 수 있다는 점을 확인할 수 있었는데 확률과 통계 발표를 준비하며 조사했던 내용 중 하나인 베이즈 분류기에 양자컴퓨터가 활용될 수 있을것이라는 생각이 들어 간단한 실습을 통해 탐구를 진행해보고자 하였다.
===================================================================================
2. 이론:
H Gate(하다마드 게이트): 고정된 Qubit인 ∣0⟩이나 ∣1⟩을 50대 50의 중첩상태로 만든다.

<img width="133" alt="image" src="https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/381a45a6-d9ad-4bd9-9cae-2123d7672327">


<img width="200" alt="image" src="https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/227df7ea-a63a-4407-94be-4e9eb4bc8308">




CX(CNOT, Controlled Not Gate) 게이트: 첫번째 아웃풋은 첫번째 인풋을 그대로 반환하고 
			두번째 아웃풋은 첫번째 인풋이 1이라면 두번째 인풋에 X Gate를 통과시킨 값을, 
			0이라면 그대로 통과시킨 값을 반환한다.
			양자 컴퓨터에서 한 큐비트가 다른 큐비트에 영향을 끼치는 얽힘 상태를 볼 수 있다.
			첫번째 게이트는 두번째 게이트의 작동 유무를 결정한다.
![image](https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/e5ec88a7-f3a1-4628-9cc2-9b5964ef77e4)


Oracle: 오라클은 어떠한 입력이든 받을 수 있지만 오직 0과 1만 반환하는 함수이다. Boolean과 같다고 볼 수 있다.
결과까지의 처리 과정에 대해서 모르므로 블랙 박스라고도 부른다.
<img width="82" alt="image" src="https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/d5002e94-6652-4071-b3e0-a1f0ebc5fbc5">


U Gate: 3개의 각도를 가지는 단일 큐비트 회전 게이트이다.
<img width="230" alt="image" src="https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/e8b4e4a6-5296-430f-bc04-631e3e3fcb0a">




예: <img width="178" alt="image" src="https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/2c852256-db22-479a-858b-9580a079f12a">



__________________________________________________________________________________________________________________________________
베이즈 정리:  두 확률 변수의 사전 확률과 사후 확률 사이의 관계를 나타내는 정리이다.
	사전 확률은 베이즈 추론에서 관측자가 관측을 하기 전에 가지고 있는 확률 분포를 의미하며 
	베이즈 정리에 의해 사후 확률을 구할 수 있으며 역확률을 구할수도 있다.

<img width="161" alt="image" src="https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/70084202-2ae5-4dc9-8172-9066da8c1137">








양자컴퓨터를 활용한 베이즈 네트워크: 베이즈 네트워크란 랜덤 변수의 집합과 방향성 비순환 그래프를 통하여 
	그 집합을 조건부 독립으로 표현하는 확률의 그래픽 모델로 예를 들면 
	증상이 주어졌을때, 네트워크는 다양한 질병의 존재 확률을 계산할 수 있다. 
	복잡한 결합 분포보다 직접적인 의존성과 지연분포를 사람이 이해하는데 직관적인 장점이 있다.

![image](https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/a703956a-5a88-433a-943e-f9601c5c3980)

<img width="416" alt="image" src="https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/5569be12-e14f-4968-a12a-2cdedf1bd09d">




									(베이즈 네트워크를 구현한 양자 회로)
기각 샘플링(Rejection Sampling): 주어진 확률 분포에서 효율적으로 샘플을 생성하기 위해 이용되는 알고리즘
	주어진 확률 분포의 확률 밀도 함수를 알고 있지만 직접 샘플을 생성하기 어려울때 활용할 수 있다.
===================================================================================
3. 실험 과정:
 (환경) Python 3.10.6, {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': None, 'qiskit-finance': None, 'qiskit-optimization': None, 'qiskit-machine-learning': '0.6.1'}

Qiskit: Qiskit은 IBM에서 개발한 언어로, 클라우드 시스템을 통해 양자 컴퓨터에 접근하고 
	회로를 설계하기 위해 만들어진 오픈소스 퀀텀 컴퓨팅 프레임워크이다. 회로를 설계하여 	
	IBM의 서버로 보내면 양자컴퓨터나 시뮬레이터로 실험을 해주고 결과를 
	다시 개인에게 보내는 방식이다. 
#설치 과정
%pip install qiskit qiskit[visualization] latex qiskit_machine_learning
#%pip install git+https://github.com/qiskit-community/qiskit-textbook.git#subdirectory=qiskit-textbook-src
#양자 회로 구성
from qiskit import QuantumCircuit, execute, Aer
#3개 큐비트의 양자 레지스터에 작용하는 양자 회로 생성
circuit = QuantumCircuit(2, 2)
# 큐비트0에 H Gate(하다마드 게이트) 추가
circuit.h(0)
# 컨트롤 큐비트 0과 타켓 큐비트 1에 CX 게이트 추가
circuit.cx(0, 1)
#양자 측정을 고전 비트에 매핑
circuit.measure([0,1], [0,1])

#matplotlib로 측정한 결과를 표시
circuit.draw(output='mpl', justify='none', initial_state=True)

<img width="216" alt="image" src="https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/43e3a56c-737d-4da6-905b-b060da3b58de">


대략 1:1임을 확인할 수 있음


![image](https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/11a82e56-a0e3-439e-b137-ff9ca932d15d)
























#Rotation Gate를 통한 게이트 구현(실험에 사용한 Byskit 라이브러리에 포함)
#오라클
def oracle(circ):
    """
    Implements an oracle that flips the sign of states that contain P = 1.
    """
    circ.cu(pi, pi, 0, 0, net[0], net[1])
    circ.cu(pi, pi, 0, 0, net[0], net[1])    
    return circ
#U게이트
def u_gate(circ):
    """
    Implements the U gate that flips states about the average amplitude.
    """
    # Implements the quantum circuit that converts ψ -> |000...0>
    circ.u(-1*probToAngle(0.35), 0, 0, net[0])
    circ.u(-1*probToAngle(0.76), 0, 0, net[1])
    circ.u(-1*probToAngle(0.39), 0, 0, net[2])
    # Flipping the |000...0> state using a triple controlled Z gate condtioned on P, E and H, 
    # and applied to the ancilla
    circ.x(net)
    circ.cu(pi/4, pi, 0, 0, net[0], net[3])
    circ.cx(net[0], net[1])
    circ.cu(-pi/4, pi, 0, 0, net[1], net[3])
    circ.cx(net[0], net[1])
    circ.cu(pi/4, pi, 0, 0, net[1], net[3])
    circ.cx(net[1], net[2])
    circ.cu(-pi/4, pi, 0, 0, net[2], net[3])
    circ.cx(net[0], net[2])
    circ.cu(pi/4, pi, 0, 0, net[2], net[3])
    circ.cx(net[1], net[2])
    circ.cu(-pi/4, pi, 0, 0, net[2], net[3])
    circ.cx(net[0], net[2])
    circ.cu(pi/4, pi, 0, 0, net[2], net[3])
    circ.x(net)
    # Implements the quantum circuit that converts |000...0> -> ψ 
    circ.u(probToAngle(0.35), 0, 0, net[0])
    circ.u(probToAngle(0.76), 0, 0, net[1])
    circ.u(probToAngle(0.39), 0, 0, net[2])
    return circ

![image](https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/f535349d-756c-492c-a377-bf4b296451c5)


__________________________________________________________________________________________________________________________________
실험: 
#Byskit 예제(양자컴퓨팅 베이즈 네트워크를 구현한 라이브러리에서 예시로 제시한 코드)
#https://github.com/mlvqc/Byskit
#Byskit.py는 지나치게 길어 보고서에 작성하지 않음, 위 로테이션 게이트가 적용되있음
from qiskit import IBMQ
IBMQ.load_account()
from qiskit import Aer
backend = Aer.get_backend('qasm_simulator')
network = {'root':2,'child-1':3,'child-2':3}
loaded_net = gen_random_net(network)
b = byskit(backend, network, loaded_net)
b.plot()
evidence = {
    'one':{
        'n':1,
        'state':'1'
    },
    'two':{
        'n':5,
        'state':'0'
    }
}
sample_list = b.rejection_sampling(evidence, shots=1000, amplitude_amplification=True)
observations = {
    'one':{
        'n':2,
        'state':'0'
    },
    'two': {
        'n': 4,
        'state': '1'
    }
}
prob = b.evaluate(sample_list, observations)'''
실행 결과: 1000개의 랜덤 생성된 데이터 중 25개가 사용, 975개가 거절됨 
	구한 확률은 28%

![image](https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/e320f60b-8ecb-44d2-8254-00ca9e719c4e)













__________________________________________________________________________________________________________________________________

부채붓꽃의 꽃잎 길이와 너비를 활용해 
부채붓꽃의 관측된 꽃잎이 특정한 길이와 너비를 가질 때 부채붓꽃일 확률을 구함

#scikit learn datasets 의 붓꽃 데이터세트 사용
from sklearn.datasets import load_iris
import pandas as pd
listA = []
listB = []
iris = load_iris()
df = pd.DataFrame(data= iris.data ,  columns= ['sepal length', 'sepal width', 'petal length', 'petal width'])
df['target'] = iris.target
#라이브러리가 정수만 지원하여 수를 일정하게 키운 후 가까운 정수만 추출
for a in range(len(iris.data)):
    if iris.target[a] == 0:
        n = int(round(df['petal width'][a]*5))
        if n <= 1:
            n = 1
        if n not in listA:
            listA.append(n)
        listA = sorted(listA)
        n = int(round(df['petal length'][a]*5))
        if n < 6:
            n = 6
        if n > 8:
            n = 8
        if n not in listB:
            listB.append(n)
        listB = sorted(listB)
        
print(listA, listB)
df
(데이터 분포) 빨간색이 부채붓꽃(0), 초록색이 아이리스 버시칼라(1), 파란색이 아이리스 버지니카(3)이다.
listA == [1.2.3]
listB == [6,7,8]


<img width="215" alt="image" src="https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/b805d5ec-1408-4613-a58f-711a098df53e">
![image](https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/b1b18514-bd01-4a0b-9fc3-fd00da396242)
![image](https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/c790e513-b5e2-4713-8229-6925514ca0ed)






import byskit
backend = Aer.get_backend('qasm_simulator')
network = {'root':2,'child-1':3,'child-2':3}
loaded_net = byskit.gen_random_net(network)
b = byskit.byskit(backend, network, loaded_net)
b.plot()
#listA와 listB는 각각 꽃잎 너비와 길이로 위 붓꽃 데이터셋에서 구한 리스트이다.
#리스트를 처리하기 위해 byskit의 입력 부분의 반복문을 약간 수정했다.
evidence = {
    'one':{
        'n':listA,#꽃잎 너비
        'state':'1'
    },
    'two':{
        'n':listB,#꽃잎 길이
        'state':'0'
    }
}
print(listA, listB)
#b.rejection_sampling(evidence,amplitude_amplification=True)
sample_list = b.rejection_sampling(evidence, shots=1000,amplitude_amplification=False)
while True:
    observations = {
        'one':{
            'n':[int(input("꽃잎의 길이:"))],
            'state':'0'
        },
        'two':{
            'n':[int(input("꽃잎의 너비:"))],
            'state':'1'
        }
    }
    prob = b.evaluate(sample_list, observations)


===================================================================================

4. 실험 결과

세로축: 길이, 가로축: 너비
(10 이상은 오류로 인해 불가능하며 임의의 가중치로 인해 실행 결과는 매 실행마다 약간의 오차가 있음에 주의)





Rej:97.9	1	2	3	4	5	6	7	8	9
1	0	0	0	0	0	0	0	0	0
2	0	0	0	0	0	0	0	0	0
3	0	0	0	0	0	0	0	0	0
4	0	0	0	0	0	0	0	0	0
5	0	0	0	0	0	0	0	0	0
6	1	1	1	1	1	0	0	0	0
7	1	1	1	1	1	0	0	0	0
8	1	1	1	1	1	0	0	0	0
9	0.0952	0.0952	0.0952	0.0952	0.0952	0	0	0	0<img width="485" alt="image" src="https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/86322501-f7f2-4c96-903a-a558a9e33216">













<img width="385" alt="image" src="https://github.com/HansolSon1113/myQiskitPlayground/assets/60804796/9c88e9ba-2b93-4e24-8732-ef015d691167">














97.9%라는 많은 샘플들(기각 샘플링 후 생성된)이 거부되었지만 진폭 증폭을 통해 효율을 높일 수 있다.
베이즈 네트워크를 양자 하드웨어에 구현하면 Grover 알고리즘을 기반으로 하는 진폭 증폭의 원리로
거부 샘플링 속도를 제곱근으로 높일 수 있다. 진폭 증폭 작용을 활용하면 증거 단계의 수용된 확률이 1로 수렴하고 허용된 샘플 수를 늘릴 수 있다. 이렇게 하면 계산의 정확도를 유지할 뿐만 아니라 충분히 좋은 답을 얻기 위해 샘플링해야 하는 총 횟수를 줄인다.
양자컴퓨팅이 적용되었을때 의약품 설계, 생물학 분석, 금융 시장 모델링 등의 많은 분야에서 속도 향상을 통한 
이점을 얻을 수 있을 것으로 전망된다.

===================================================================================

5. 느낀점: 라이브러리에서 정수만 지원하여 소수로 기록된 데이터들에 적당한 수를 곱하고 가까운 정수로 변환해 활용하였는데 이로 인해 부정확해졌다는 점과 시간상 관련된 부분을 전부 공부하고 진행하지 못했다는 점에서
아쉬움이 있었다. 또한 3년전에 만들어진 라이브러리였는데 qiskit이 그동안 변하면서 일부 요소들이 변경되고 사라져 오류가 발생하고 사용할 수가 없어서 이것을 사용할 수 있도록 자료를 찾아가며 수정하고 붓꽃 데이터셋 분석을 위해 이 라이브러리를 적용하고자 리스트로 입력할 수 있도록 수정해야했다.
이렇게 라이브러리를 수정하고 양자 게이트를 이해하고자 노력하는 과정에 더해 참고할 수 있는 자료가 적어서
그동안 진행했던 다른 과제들보다 훨씬 어려웠던 것 같다.
하지만 계속해서 이해하고자 노력하고 시도했더니 목표를 달성하는데 성공했고 매우 큰 성취감을 느낄 수 있었다.
다음에는 빠르게 진행하기 위해 간단하게 알아보고 넘어갔던 원리들과 알고리즘들에 대해 자세히 알아보고 싶다.
