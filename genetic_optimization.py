import pickle
import random
import operator
import numpy as np

K=10
lb=0.05
ub=0.25
hof_ls=[]

fr=open("./covar.pkl","rb")
covar=pickle.load(fr)

fr=open("./mean.pkl","rb")
mean=pickle.load(fr)

fr=open("./code.pkl","rb")
main_ls=pickle.load(fr)

'''
fr=open("./mean_stat.pkl","rb")
mean_stat=pickle.load(fr)

fr=open("./var_stat.pkl","rb")
var_stat=pickle.load(fr)
'''

# evalutaion
def fitness(population, weight):
    fit = [0]*len(population) # (mean,var)
    length = len(population[0])
    for idx,(individual,w) in enumerate(zip(population,weight)):
        if(sum(individual)>K):
            rand_K=random.randint(1,K)
            up_list=[i for i,x in enumerate(individual) if x==1]
            dic_up={}
            for j in up_list:
                dic_up[j]=w[j]

            sorted_up = sorted(dic_up.items(), key=operator.itemgetter(1))
            up_idx=[]
            for j in sorted_up:
                up_idx.append(j[0])

            for j in up_idx[0:sum(individual)-rand_K]:
                individual[j] = 0

        x=[0]*length
        div=0

        for i in range(length):
            div+=individual[i]*w[i]

        if(div!=0):
            for i in range(length):
                x[i]=lb*individual[i]+(w[i]*individual[i])/div*(1-lb*sum(individual))

        pf_return=np.matmul(mean,x)
        pf_variance=np.matmul(np.matmul(x,covar),x)
        fit[idx]=( pf_return ,pf_variance)

    return fit

# selection
def selection_tournament(population,weight,fit,size=2,mode=True):
    length=len(population)
    temp_pop=[]
    temp_weight=[]

    if(mode):
        for i in range(length):
            chosen = []
            chosen_idx=[]
            aspirants = random.sample(range(0,length), size)
            for j in aspirants:
                chosen_idx.append(j)
                chosen.append(fit[j][0] - fit[j][1])

            max_idx = chosen.index(max(chosen))

            temp_pop.append(population[chosen_idx[max_idx]])
            temp_weight.append(weight[chosen_idx[max_idx]])


    return temp_pop,temp_weight

def selection_NSGA2(population, weight, fit):

    pop_length=len(population)
    dominant_ls={}
    for i in range(pop_length):
        dominant_ls[i]=[]

    dominated_ls=[0]*pop_length

    for idx1,f1 in enumerate(fit):
        for idx2, f2 in enumerate(fit):
            if(idx1<idx2):
                if((f1[0]>f2[0] and f1[1]<=f2[1]) or (f1[0]>=f2[0] and f1[1]<f2[1])):
                    dominant_ls[idx1].append(idx2)
                    dominated_ls[idx2] += 1

                if((f1[0]<f2[0] and f1[1]>=f2[1]) or (f1[0]<=f2[0] and f1[1]>f2[1])):
                    dominant_ls[idx2].append(idx1)
                    dominated_ls[idx1] += 1

    remain_num=pop_length//2
    result_pop=[]
    result_w=[]

    while(remain_num>0):
        temp_ls= [i for i, x in enumerate(dominated_ls) if x == 0]
        if(len(temp_ls)>remain_num):
            distance={}
            for n in range(2):
                crowd=[]
                for j in temp_ls:
                    crowd.append((j,fit[j][n]))
                crowd.sort(key=lambda element:element[1])
                distance[crowd[0][0]]=float("inf")
                distance[crowd[-1][0]] = float("inf")
                if(crowd[-1][1] == crowd[0][1]):
                    continue

                norm=2*float(crowd[-1][1] - crowd[0][1])
                for prev, cur, nxt in zip(crowd[:-2],crowd[1:-1],crowd[2:]):
                    if(n==0):
                        distance[cur[0]] = (nxt[1]-prev[1])/norm
                    else:
                        distance[cur[0]] += (nxt[1]-prev[1])/norm

            sorted_up = sorted(distance.items(), key=operator.itemgetter(1),reverse=True)
            crowd_idx = []
            for j in sorted_up:
                crowd_idx.append(j[0])

            for j in crowd_idx[0:remain_num]:
                result_pop.append(population[j])
                result_w.append(weight[j])

            break
        else:
            for j in temp_ls:
                result_pop.append(population[j])
                result_w.append(weight[j])
                for k in dominant_ls[j]:
                    dominated_ls[k]-=1
                dominated_ls[j] -= 1

            remain_num-=len(temp_ls)

    return result_pop, result_w

# cross_over
def cross_cover(population,weight,cxpb=0.9):
    length = len(population[0])
    exc_ls=list(set(list(range(len(population))))-set(hof_ls))

    for i in range((len(population)-1)//2):
        idx1, idx2 = random.sample(exc_ls, 2)
        while(idx1==idx2):
            idx1,idx2=random.sample(exc_ls , 2)

        exc_ls.remove(idx1)
        exc_ls.remove(idx2)

        if(random.random()<cxpb):
            cxpoint1 = random.randint(1,length-1)
            cxpoint2 = random.randint(1,length-1)

            while(cxpoint1==cxpoint2):
                cxpoint2 = random.randint(1, length - 1)

            if(cxpoint1>cxpoint2):
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1

            population[idx1][cxpoint1:cxpoint2], population[idx2][cxpoint1:cxpoint2]=population[idx2][cxpoint1:cxpoint2], population[idx1][cxpoint1:cxpoint2]
            weight[idx1][cxpoint1:cxpoint2], weight[idx2][cxpoint1:cxpoint2] = weight[idx2][cxpoint1:cxpoint2], weight[idx1][cxpoint1:cxpoint2]

    return population, weight


def mutation(population,weight,mutpb=1.0,indpb=1.0, sigma=0.15):
    length=len(population[0])
    exc_ls = list(set(list(range(length))) - set(hof_ls))

    for k in exc_ls:
        if(random.random()<mutpb):
            for i in range(length):
                if(random.random()<1/length):
                    population[k][i] = 1

                if (random.random() < indpb):
                    gauss=random.gauss(0.0,sigma)

                    while(weight[k][i]+gauss>1.0 or weight[k][i]+gauss<0.0):
                        gauss = random.gauss(0.0, sigma)

                    weight[k][i]+=gauss

    return population, weight


def halloffame(fit,size=1):
    fit_val=[]
    fit_val_primal=[]
    for v1,v2 in fit:
        fit_val.append(v1-v2)
        fit_val_primal.append(v1-v2)

    fit_val.sort(reverse=True)
    idx_ls=[]
    cnt=0
    for val in fit_val:
        if(cnt>size-1):
            break
        idx_ls.append(fit_val_primal.index(val))
        cnt+=1

    return idx_ls


def select_best(population,weight,fit):
    rank=[fit[i][0]-fit[i][1] for i in range(len(population))]
    max_idx=rank.index(max(rank))
    return population[max_idx],weight[max_idx],fit[max_idx]

def select_dominant(population,weight,fit):
    return


if __name__ == "__main__":

    random.seed(64)
    individual_size = len(main_ls)
    population_size=250
    N_gen=100
    population=[]
    weight=[]

    # creator pop & ind
    for _ in range(population_size):
        temp=0
        randK=random.randint(1,K)
        individual = [0] * individual_size
        for i in range(individual_size):
            rand=random.randint(0,1)
            if(rand==1):
                temp+=1
                individual[i]=1
                if(temp==randK):
                    break

        w=[random.random() for _ in range(individual_size)]
        population.append(individual)
        weight.append(w)

    for iter in range(N_gen):
        fit=fitness(population, weight)
        hof_ls=halloffame(fit)  # reproduction
        print("------- episode: " + str(iter + 1) + "--------")
        print(population)
        print(weight)
        print(fit)
        offspring_population, offspring_weight = selection_tournament(population, weight, fit)
        offspring_fit = fitness(offspring_population, offspring_weight)
        population, weight = selection_NSGA2(population+offspring_population, weight+offspring_weight, fit+offspring_fit)
        population, weight = cross_cover(population, weight)
        population, weight = mutation(population, weight)

    fit = fitness(population, weight)
    ind, w, f = select_best(population, weight, fit)

    print("delta :", ind)
    print("weight :", w)
    print("fit :", f)

