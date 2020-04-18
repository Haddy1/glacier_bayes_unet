class Preprocessor:
    filter_list = []

    def add_filter(self, filter):
        self.filter_list.append(filter)

    def process(self, imgs):

        imgs_processed = []
        for img in imgs:
            for filter in self.filter_list:
                img = filter(img)

            imgs_processed.append(img)

        return imgs_processed
