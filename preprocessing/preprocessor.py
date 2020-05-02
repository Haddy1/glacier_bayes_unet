class Preprocessor:
    filter_list = []

    def add_filter(self, filter):
        self.filter_list.append(filter)

    def process(self, img):

        for filter in self.filter_list:
            img= filter(img)


        return img


