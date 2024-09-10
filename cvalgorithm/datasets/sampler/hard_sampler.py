class SortingDataset(InferDataset):
    def __init__(self, images_data_info, template_folder, pcs_roi_info, logger, csv_data_list, template_size,
                 template_size_dual, panel_infer=False, **kwargs):
        self.template_folder = template_folder
        self.pcs_roi_info = pcs_roi_info
        self.logger = logger if logger is not None else print
        self.template_size = template_size
        self.template_size_dual = template_size_dual
        self.date_time_list = []
        self.panel_infer = panel_infer
        self.sn_pcs_set = set()

        used_data = set()
        for csv_data in csv_data_list[1:]:
            sn = csv_data[1]
            pcs = csv_data[2]
            sn_cps = f"{sn}_{pcs}"
            used_data.add(sn_cps)
        if len(used_data) > 1:
            used_data.remove(sn_cps)
        self.used_data = used_data
        super().__init__(images_data_info, **kwargs)

    def load_data(self):
        self.logger("开始处理数据")
        date_folder = [os.path.join(self.images_data_info, f) for f in os.listdir(self.images_data_info)]
        for date_time in date_folder:
            if not os.path.isdir(date_time):
                continue
            self.date_time_list += [os.path.join(date_time, f) for f in os.listdir(date_time)]

        for date_time in tqdm(self.date_time_list):
            if not os.path.isdir(date_time):
                continue
            image_paths = ls_all_files(date_time, postfix=IMAGE_POSTFIX, logger=self.logger)
            for image_path in image_paths:
                # panel推理不或选择pcs进行推理
                if self.panel_infer and "panel" not in image_path:
                    continue
                # pcs推理不会选择panel进行推理
                if not self.panel_infer and "panel" in image_path:
                    continue
                try:
                    image_name = os.path.basename(image_path)
                    info = image_name.split("-")
                    sn = info[0]
                    pcs_id = int(info[2][3:]) if not self.panel_infer else 'panel'
                    sn_pcs_id = f"{sn}_{pcs_id}"
                    self.sn_pcs_set.add(sn_pcs_id)
                    if self.used_data is not None and sn_pcs_id in self.used_data:
                        self.logger(f"Skip sn {sn_pcs_id}, that has already been inferred infered.")
                        continue

                    side = info[1]
                    self.image_list.append(
                        dict(
                            image_paths=image_path,
                            image_name=image_name,
                            date_time=os.path.basename(os.path.dirname(date_time)),
                            sn_id=sn,
                            side=side,
                            pcs_id=pcs_id,
                            node_path=[],
                            detail_infos=dict()
                        )
                    )
                except Exception as error:
                    self.logger(f"load data {image_path} error, {str(error)}")

    def __getitem__(self, item):
        image_infos = self.image_list[item]
        image_path = image_infos['image_paths']
        side = image_infos['side']
        sn_id = image_infos['sn_id']
        pcs_id = image_infos['pcs_id']
        is_dual = False
        if pcs_id in self.pcs_roi_info[side]['orig']['pcs']:
            rois = self.pcs_roi_info[side]['orig']['roi']
        else:
            rois = self.pcs_roi_info[side]['dual']['roi']
            is_dual = True
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            h, w = self.template_size
            image = cv2.resize(image, (w, h))
            count = 0
            image_infos_list = []
            for roi in rois:
                x1, y1, w, h = roi
                crop_image = image[y1:y1 + h, x1:x1 + w]
                if self.template_folder is not None and os.path.exists(self.template_folder):
                    template_name = f"{side}-{count}-dual.jpg" if is_dual else f"{side}-{count}.jpg"
                    template_image = cv2.imread(os.path.join(self.template_folder, template_name))
                    images = [crop_image, template_image]
                else:
                    images = [crop_image]
                count = count + 1
                h, w = images[0].shape[:2]
                h = max(min(h, MAX_SIZE), MIN_SIZE)
                w = max(min(w, MAX_SIZE), MIN_SIZE)
                images = [cv2.resize(img, (w, h)) for img in images]
                images = [torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() for img in images]

                infos = dict()
                infos['images'] = images
                infos['sn_id'] = sn_id
                infos['pcs_id'] = pcs_id
                infos['side'] = side
                infos['date_time'] = image_infos['date_time']
                infos['count'] = count
                infos['image_name'] = os.path.join(os.path.basename(image_path))
                infos['image_path'] = image_path
                infos['org_shape'] = (h, w)
                infos['node_path'] = []
                infos['detail_infos'] = dict()
                image_infos_list.append(infos)
        except Exception as error:
            self.logger(f"加载数据{image_path}失败!")
            traceback.print_exc()
            image_infos['images'] = None
            image_infos['image_path'] = image_path
            return image_infos

        return image_infos_list

    def collate_data(self):
        batch_data = []
        for _ in range(self.batch_size):
            if self.collate_count < len(self.indices):
                data = self.image_queue.get()
                if isinstance(data, dict):
                    data['images'] = [img.cuda() for img in data['images']] if data['images'] is not None else data[
                        'images']
                    batch_data.append(data)
                elif isinstance(data, list):
                    for d in data:
                        d['images'] = [img.cuda() for img in d['images']] if d['images'] is not None else d['images']
                        batch_data.append(d)

                self.collate_count += 1
                self._try_put_index()

        return batch_data