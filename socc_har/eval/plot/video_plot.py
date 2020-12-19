import os
from PIL import ImageDraw, ImageFont
import base64
from tqdm.auto import tqdm
from torchvision.transforms import functional as T
from torchvision import io
from pytorch_lightning.loggers import LightningLoggerBase
from IPython.core.display import HTML


class VideoPlot:

    def __init__(self, logger: LightningLoggerBase, video_path: str, segment: [int], classes: [str], match_key: str,
                 ground_truth_annotations: dict, prediction_annotations: dict, save_dir='.'):
        self.classes = classes
        self.segment = segment
        self.logger = logger
        self.video_path = video_path
        self.ground_truth_annotations = ground_truth_annotations
        self.prediction_annotations = prediction_annotations
        self.save_dir = save_dir
        self.match_key = match_key
        self.title = f'{self.match_key}@{self.segment[0]}-{self.segment[1]}'

    def decode(self):

        path = f'{self.save_dir}/demo-{self.title}.mp4'
        if os.path.exists(path):
            return

        src_video, _, _ = io.read_video(self.video_path, self.segment[0], self.segment[1], pts_unit='sec')
        duration = self.segment[1] - self.segment[0]
        num_frames = src_video.shape[0]
        fps = num_frames / duration

        demo_vid = torch.zeros((num_frames, 360, 640, 3))

        for idx, img in tqdm(enumerate(src_video), total=src_video.shape[0]):
            # img: [H, W, C]

            img = img.permute((2, 0, 1))
            img = T.to_pil_image(img)

            img = T.resize(img, (360, 640))

            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            w, h = font.getsize(self.title)
            draw.rectangle((0, 10, w, 10 + h), fill='white')
            draw.text((0, 10), self.title, (0, 0, 0))

            offset = self.segment[0] + round(idx/fps)

            pred_labels = [anno['label'] for anno in self.prediction_annotations if ClassificationDataset.overlap(anno['segment'], [offset, offset+1]) > 0]
            pred_labels.sort()
            gt_labels = [anno['label'] for anno in self.ground_truth_annotations if ClassificationDataset.overlap(anno['segment'], [offset, offset+1]) > 0]
            gt_labels.sort()

            predicted_line = f'predicted: {", ".join(pred_labels)}'
            w, h = font.getsize(predicted_line)
            draw.rectangle((0, 20, w, 20 + h), fill='white')
            draw.text((0, 20), predicted_line, (0, 0, 0))

            correct = ''.join(pred_labels) == ''.join(gt_labels)
            if correct:
                gt_line = f'correct!'
            else:
                gt_line = f'should be: {", ".join(gt_labels)}'
            w, h = font.getsize(gt_line)
            draw.rectangle((0, 30, w, 30 + h), fill='white')
            draw.text((0, 30), gt_line, (0, 0, 0))

            img = T.to_tensor(img)
            img = img.permute((1,2,0))


            demo_vid[idx] = img

        res = io.write_video(path, (demo_vid * 255).type(torch.uint8), 25)
        return demo_vid

    def show(self):
        self.decode()
        location = f'{self.save_dir}/demo-{self.title}.mp4'
        if os.path.isfile(location):
            ext = 'mp4'
        else:
            print("Error: Please check the path.")
            return
        video_encoded = open(location, "rb")
        binary_file_data = video_encoded.read()
        base64_encoded_data = base64.b64encode(binary_file_data)
        video_tag = '<video width="320" height="240" controls alt="test" src="data:video/{0};base64,{1}">'.format(ext, base64_encoded_data)
        return HTML(data=video_tag)
        #print(location)
        #return Video('/content/develop/Classifier/notebooks/demo-266236@2@4576-4606.mp4')

    def save(self):
        self.decode()
        filename = f'{self.save_dir}/demo-{self.title}.mp4'
        with self.logger.experiment.context_manager("test"):
            self.logger.experiment.log_asset(filename)