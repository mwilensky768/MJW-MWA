def apply_chisq_test(self, INS, ES=None, event_record=False):
        """
        Calculates the p-value of each shape and channel using chisq_test().
        Should the p-value of a shape or channel be less than the significance
        threshold, alpha, the entire observation will be flagged for that shape
        or channel.
        """
        if event_record and (ES is None):
            ES = ES()

        p_min = 0
        while p_min < self.alpha:
            p_min, shape_min, f_point = self.chisq_test()
            if p_min < self.alpha:
                if shape_min is 'point':
                    event = (0, 0, slice(f_point, f_point + 1), 'point')
                    INS.data[:, 0, slice(f_point, f_point + 1)] = np.ma.masked
                    if event_record:
                        ES.chisq_events.append(event)
                else:
                    event = (0, 0, self.slice_dict[shape_min], shape_min)
                    INS.data[:, 0, self.slice_dict[shape_min]] = np.ma.masked
                    if event_record:
                        ES.chisq_events.append(event)

                INS.data_ms[:, event[2]] = INS.mean_subtract(f=event[2])

        return(ES)

        def chisq_test(self, INS):

            """
            A test to measure the chi-square of the binned shapes and channels
            relative to standard normal noise (the null hypothesis of the filter).
            """

            p_min = 1
            shape_min = None
            for shape in self.slice_dict:
                if shape is 'point':
                    p = 1
                    f_point = None
                    for f in range(INS.metric_array.shape[2]):
                        stat, p_point = util.chisq(*ES.hist_make(event=(0, 0, slice(f, f + 1))))
                        if p_point < p:
                            p = p_point
                            f_point = f
                else:
                    stat, p = util.chisq(*ES.hist_make(event=(0, 0, self.slice_dict[shape])))
                if p < p_min:
                    p_min = p
                    shape_min = shape

            return(p_min, shape_min, f_point)
