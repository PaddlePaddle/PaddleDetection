class DetectorSOLOv2(Detector):
    """
    Args:
        config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        use_gpu (bool): whether use gpu
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
        threshold (float): threshold to reserve the result for output.
    """

    def __init__(self,
                 pred_config,
                 model_dir,
                 use_gpu=False,
                 run_mode='fluid',
                 threshold=0.5):
        self.pred_config = pred_config
        self.predictor = load_predictor(
            model_dir,
            run_mode=run_mode,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            use_gpu=use_gpu)

    def predict(self,
                image,
                threshold=0.5,
                warmup=0,
                repeats=1,
                run_benchmark=False):
        '''
        Args:
            image (str/np.ndarray): path of image/ np.ndarray read by cv2
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): 'segm': np.ndarray,shape:[N, im_h, im_w]
                            'cate_label': label of segm, shape:[N]
                            'cate_score': confidence score of segm, shape:[N]
        '''
        inputs = self.preprocess(image)
        np_label, np_score, np_segms = None, None, None
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            np_label = self.predictor.get_output_handle(output_names[
                1]).copy_to_cpu()
            np_score = self.predictor.get_output_handle(output_names[
                2]).copy_to_cpu()
            np_segms = self.predictor.get_output_handle(output_names[
                3]).copy_to_cpu()

        t1 = time.time()
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            np_label = self.predictor.get_output_handle(output_names[
                1]).copy_to_cpu()
            np_score = self.predictor.get_output_handle(output_names[
                2]).copy_to_cpu()
            np_segms = self.predictor.get_output_handle(output_names[
                3]).copy_to_cpu()
        t2 = time.time()
        ms = (t2 - t1) * 1000.0 / repeats
        print("Inference: {} ms per batch image".format(ms))

        # do not perform postprocess in benchmark mode
        results = []
        if not run_benchmark:
            return dict(segm=np_segms, label=np_label, score=np_score)
        return results