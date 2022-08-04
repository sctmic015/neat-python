class anglequee:
    def __init__(self):
        self.count = 0
        self.first = []
        self.second = []
        self.third = []
        self.fourth = []

    def add(self, x):
        if self.count == 0:
            self.first = x
            self.second = x
            self.third = x
            self.fourth = x
            self.count += 1
        else:
            self.fourth = self.third
            self.third = self.second
            self.second = self.first
            self.first = x

    def get_moving_average(self):
        moving_average = [(x1 + x2 + x3 + x4) / 4 for x1, x2, x3, x4 in
                          zip(self.first, self.second, self.third, self.fourth)]
        return moving_average


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    aq = anglequee()
    aq.add(x)
    print(aq.get_moving_average())
    x1 = [2, 2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2, 2]
    aq.add(x1)
    print(aq.get_moving_average())
    x2 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    aq.add(x2)
    print(aq.get_moving_average())
    x3 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    aq.add(x3)
    print(aq.get_moving_average())