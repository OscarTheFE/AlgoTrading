# update mean without overflow
def enqueue_mean(original_mean, size, new_item):
    return original_mean + (new_item - original_mean) / (size + 1)


def dequeue_mean(original_mean, size, old_item):
    if size == 0:
        return 0
    if size == 1:
        return 0
    return size / (size - 1) * original_mean - old_item / (size - 1)


class CircularPairQueue(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue_x = [None] * capacity
        self.queue_y = [None] * capacity
        self.tail = -1
        self.head = 0
        self.size = 0

        self.x_mean = 0
        self.xlag_mean = 0
        self.y_mean = 0
        self.ylag_mean = 0
        self.xx_mean = 0
        self.yy_mean = 0
        self.xy_mean = 0

        self.xxlag_mean = 0
        self.yylag_mean = 0
        self.xlagxlag_mean = 0
        self.ylagylag_mean = 0
        self.xylag_mean = 0
        self.yxlag_mean = 0
        self.ylagxlag_mean = 0

    def get_size(self):
        return self.size

    def dequeue(self):
        if self.size == 0:
            print("no item to dequeue")
            return
        else:
            temp_x = self.queue_x[self.head]
            temp_y = self.queue_y[self.head]
            self.queue_x[self.head] = None
            self.queue_y[self.head] = None
            self.head = (self.head + 1) % self.capacity

            self.x_mean = dequeue_mean(self.x_mean, self.size, temp_x)
            self.y_mean = dequeue_mean(self.y_mean, self.size, temp_y)
            self.xx_mean = dequeue_mean(self.xx_mean, self.size, temp_x * temp_x)
            self.yy_mean = dequeue_mean(self.yy_mean, self.size, temp_y * temp_y)
            self.xy_mean = dequeue_mean(self.xy_mean, self.size, temp_x * temp_y)
            self.xlag_mean = dequeue_mean(self.xlag_mean, self.size - 1, temp_x)
            self.ylag_mean = dequeue_mean(self.ylag_mean, self.size - 1, temp_y)

            self.xxlag_mean = dequeue_mean(self.xxlag_mean, self.size - 1, temp_x * self.queue_x[self.head])
            self.yylag_mean = dequeue_mean(self.yylag_mean, self.size - 1, temp_y * self.queue_y[self.head])
            self.xlagxlag_mean = dequeue_mean(self.xlagxlag_mean, self.size - 1, temp_x * temp_x)
            self.ylagylag_mean = dequeue_mean(self.ylagylag_mean, self.size - 1, temp_y * temp_y)
            self.xylag_mean = dequeue_mean(self.xylag_mean, self.size - 1, temp_y * self.queue_x[self.head])
            self.yxlag_mean = dequeue_mean(self.yxlag_mean, self.size - 1, temp_x * self.queue_y[self.head])
            self.ylagxlag_mean = dequeue_mean(self.ylagxlag_mean, self.size - 1, temp_y * temp_x)
            self.size -= 1
            return

    def enqueue(self, item_x, item_y):
        temp_tail = self.tail
        self.tail = (self.tail + 1) % self.capacity
        if self.size == self.capacity:
            self.dequeue()
        self.queue_x[self.tail] = item_x
        self.queue_y[self.tail] = item_y

        self.x_mean = enqueue_mean(self.x_mean, self.size, item_x)
        self.y_mean = enqueue_mean(self.y_mean, self.size, item_y)
        self.xx_mean = enqueue_mean(self.xx_mean, self.size, item_x * item_x)
        self.yy_mean = enqueue_mean(self.yy_mean, self.size, item_y * item_y)
        self.xy_mean = enqueue_mean(self.xy_mean, self.size, item_x * item_y)

        if temp_tail != -1:
            self.xlag_mean = enqueue_mean(self.xlag_mean, self.size - 1, self.queue_x[temp_tail])
            self.ylag_mean = enqueue_mean(self.ylag_mean, self.size - 1, self.queue_y[temp_tail])
            self.xlagxlag_mean = enqueue_mean(self.xlagxlag_mean, self.size - 1, self.queue_x[temp_tail] * self.queue_x[temp_tail])
            self.ylagylag_mean = enqueue_mean(self.ylagylag_mean, self.size - 1, self.queue_y[temp_tail] * self.queue_y[temp_tail])
            self.xxlag_mean = enqueue_mean(self.xxlag_mean, self.size - 1, item_x * self.queue_x[temp_tail])
            self.yylag_mean = enqueue_mean(self.yylag_mean, self.size - 1, item_y * self.queue_y[temp_tail])
            self.xylag_mean = enqueue_mean(self.xylag_mean, self.size - 1, item_x * self.queue_y[temp_tail])
            self.yxlag_mean = enqueue_mean(self.yxlag_mean, self.size - 1, item_y * self.queue_x[temp_tail])
            self.ylagxlag_mean = enqueue_mean(self.ylagxlag_mean, self.size - 1, self.queue_y[temp_tail] * self.queue_x[temp_tail])
        self.size += 1
        return

    def display(self):
        if self.tail < self.head:
            dis_x = self.queue_x[self.head:] + self.queue_x[:self.tail + 1]
            dis_y = self.queue_y[self.head:] + self.queue_y[:self.tail + 1]
            print(dis_x)
            print(dis_y)
        else:
            dis_x = self.queue_x[self.head:self.tail + 1]
            dis_y = self.queue_y[self.head:self.tail + 1]
            print(dis_x)
            print(dis_y)

    def output(self):
        print("size: ", self.size)
        print("x_mean: ", self.x_mean)
        print("y_mean: ", self.y_mean)
        print("xx_mean: ", self.xx_mean)
        print("yy_mean: ", self.yy_mean)
        print("xy_mean: ", self.xy_mean)
        print("xlag_mean: ", self.xlag_mean)
        print("ylag_mean: ", self.ylag_mean)

        print("xxlag_mean: ", self.xxlag_mean)
        print("yylag_mean: ", self.yylag_mean)
        print("xylag_mean: ", self.xylag_mean)
        print("yxlag_mean: ", self.yxlag_mean)
        print("xlagxlag_mean: ", self.xlagxlag_mean)
        print("ylagylag_mean: ", self.ylagylag_mean)
        print("ylagxlag_mean: ", self.ylagxlag_mean)
        return

if __name__ == "__main__":
    cq = CircularPairQueue(3)
    for i in range(20):
        cq.enqueue(i, i + 10)
        cq.display()
        cq.output()
        print()
