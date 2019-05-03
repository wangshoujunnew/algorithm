import numpy as np
import numpy.random as rd
import itertools
# 用户数据 1-10
user = [x for x in range(10)]
# 电影数据 1-10
movie = [x for x in range(10)]
# 用户对电影的评分矩阵 10*10
data = rd.randint(low=0,high=5,size=100)
score_v = data.reshape((10,10))


# -------------------


def lookup(index, np_data):
    """
    通过index得到用户对电影的评分
    """
    return np_data[index]

def user_like(x, y):
    """
    用户相似度计算, 使用皮尔逊相关系数计算用户的相似度
    ,这里使用余弦相似度
    从都买过的商品来计算
    [1,,2..], [...]
    1.6
    """
    # reset x,y, get same
    index_list = []
    for (a,b,c) in zip(x,y,range(len(x))):
        if None in [a,b] or 0 in [a,b]: # remove
            index_list.append(c)
    A,B = list(map(
        lambda e:np.array([i for i,pos in zip(e,range(len(e))) if pos not in index_list]),
        [x,y]
    ))
    # print('调整之后用户向量为:\n',A,B)
    num = float(np.dot(A , B.T)) #若为行向量则 
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    cos = num / denom #余弦值
    sim = 0.5 + 0.5 * cos #归一化
    return sim, A, B

def same_maxuser(target_user):
    #2指的是几个元素组合
    result = []
    
    #     iters = itertools.combinations(user,2)
    #     for f,b in iters:
    for f,b in [(target_user,x) for x in user if x != target_user]:
        S = user_like(lookup(f, score_v),lookup(b, score_v))
        result.append(
            {
                "用户1":f,
                "用户2":b,
                "相似度":S[0],
                "向量":(S[1],S[2])
            }
        )
    result.sort(key=lambda e:e['相似度'],reverse=True)
    return result[0]

def get_scoremax(index, np_data):
    """
    得到用户最感兴趣的电影,评分最大的
    """
    return sorted(zip(np_data[index],range(len(np_data[index]))),key=lambda e:e[0], reverse=True)[0][1]
    
    
    
# ----------------------------



# 输入一个用户
user1 = 2
# 得到相似度最大的用户
usermax = same_maxuser(user1)
print(usermax)
# 给他推荐的电影是
get_scoremax(usermax['用户2'], score_v)

