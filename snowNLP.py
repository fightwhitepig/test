from snownlp import SnowNLP


# 自带了一个词库，然后还可以进行情感分析，把繁体化简，把中文变成汉语拼音，
# 还可以提取关键字，提取摘要（这个感觉666），还可以进行文本分类，对文本的相似性进行判断
# 1.安装：pip3 install snownlp
# 2.基本用法：
# from snownlp import SnowNLP
# s = SnowNLP(u'一次满意的购物')
# s.words
# 1) s.words        词语
# 2) s.sentences   句子
# 3) s.sentiments 情感偏向,0-1之间的浮点数，越靠近1越积极
# 4) s.pinyin         转为拼音
# 5) s.han             转为简体
# 6) s.keywords(n) 提取关键字,n默认为5
# 7) s.summary(n)  提取摘要,n默认为5
# 8) s.tf                   计算term frequency词频
# 9) s.idf                 计算inverse document frequency逆向文件频率
# 10) s.sim(doc,index)          计算相似度
#
# 这个sim的话，后面跟的是一个可以迭代的东西。

def snow(path):
    try:
        f = open(path, 'r')
        str = f.read()
    finally:
        if f:
            f.close()
    s = SnowNLP(str)
    print(s.words)

