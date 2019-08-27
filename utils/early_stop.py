import logging
logger = logging.getLogger("Ealiy Stop")

class EarlyStop():
    BEST = 1
    CONTINUE = 2
    STOP = -1

    def __init__(self,max_retry):
        self.best_value = -9999999
        self.retry_counter = 0
        self.max_retry= max_retry

    def decide(self,value):

        if value>= self.best_value:
            logger.debug("[早停] 新F1值%f>旧F1值%f，记录最好的F1，继续训练",value,self.best_value)
            # 所有的都重置
            self.retry_counter = 0
            self.best_value = value
            return EarlyStop.BEST

        # 甭管怎样，先把计数器++
        self.retry_counter+=1
        logger.debug("[早停] 新F1值%f<旧F1值%f,早停计数器:%d", value, self.best_value,self.retry_counter)

        # 如果还没有到达最大尝试次数，那就继续
        if self.retry_counter < self.max_retry:
            logger.debug("[早停] 早停计数器%d未达到最大尝试次数%d，继续训练",self.retry_counter,self.max_retry)
            return EarlyStop.CONTINUE

        logger.debug("[早停] 早停计数器%d都已经达到最大，退出训练", self.retry_counter)
        # 如果到达最大尝试次数，并且也到达了最大decay次数
        return EarlyStop.STOP