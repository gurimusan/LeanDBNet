import numpy as np

from .eval import DetectionIoUEvaluator


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self


class QuadMetric:
    def __init__(self, is_output_polygon=False):
        self.is_output_polygon = is_output_polygon
        self.evaluator = DetectionIoUEvaluator()

    def measure(self, image, target, output, box_thresh=0.6):
        """
        Args:
            image: image as tensor (N, C, H, W)
            target: metadata(e.g. polygons, polygon_classes, ...)
            output: (polygons, ...)
        """
        results = []
        pred_polygons_batch = output[0]
        pred_scores_batch = output[1]
        for tgt, pred_polygons, pred_scores in zip(target, pred_polygons_batch, pred_scores_batch):
            polygons = tgt["polygons"]
            ignore_tags = tgt["polygon_ignores"]

            gt = [
                dict(points=np.int64(polygons[i]), ignore=ignore_tags[i])
                for i in range(len(polygons))
            ]
            if self.is_output_polygon:
                pred = [
                    dict(points=pred_polygons[i]) for i in range(len(pred_polygons))
                ]
            else:
                pred = []
                # print(pred_polygons.shape)
                for i in range(pred_polygons.shape[0]):
                    if pred_scores[i] >= box_thresh:
                        # print(pred_polygons[i,:,:].tolist())
                        pred.append(
                            dict(points=pred_polygons[i, :, :].astype(np.int32))
                        )
                # pred = [dict(points=pred_polygons[i,:,:].tolist()) if pred_scores[i] >= box_thresh for i in range(pred_polygons.shape[0])]
            results.append(self.evaluator.evaluate_image(gt, pred))
        return results

    def validate_measure(self, image, target, output, box_thresh=0.6):
        return self.measure(image, target, output, box_thresh)

    def evaluate_measure(self, image, target, output):
        return (
            self.measure(image, target, output),
            np.linspace(0, image.shape[0]).tolist(),
        )

    def gather_measure(self, raw_metrics):
        raw_metrics = [image_metrics
                       for batch_metrics in raw_metrics
                       for image_metrics in batch_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        precision.update(result["precision"], n=len(raw_metrics))
        recall.update(result["recall"], n=len(raw_metrics))
        fmeasure_score = (
            2 * precision.val * recall.val / (precision.val + recall.val + 1e-8)
        )
        fmeasure.update(fmeasure_score)

        return recall, precision, fmeasure
