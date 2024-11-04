import numpy as np
# from mmdet3d.apis import init_model, inference_multi_modality_detector
from mmdet3d.apis import MultiModalityDet3DInferencer, LidarDet3DInferencer
from mmdet3d.structures import LiDARInstance3DBoxes


class Detector:
    def __init__(self):
        # Add your initialization logic here
        pass

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the detector. The location is defined with respect to the actor center
        -- x axis is longitudinal (forward-backward)
        -- y axis is lateral (left and right)
        -- z axis is vertical
        Unit is in meters

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 1280, 'height': 720, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
                      'range': 50, 
                      'rotation_frequency': 20, 'channels': 64,
                      'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,
                      'id': 'LIDAR'},

            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'}
        ]

        return sensors

    def detect(self, sensor_data):
        """
        Add your detection logic here
            Input: sensor_data, a dictionary containing all sensor data. Key: sensor id. Value: tuple of frame id and data. For example
                'Right' : (frame_id, numpy.ndarray)
                    The RGBA image, shape (H, W, 4)
                'Left' : (frame_id, numpy.ndarray)
                    The RGBA image, shape (H, W, 4)
                'LIDAR' : (frame_id, numpy.ndarray)
                    The lidar data, shape (N, 4)
            Output: a dictionary of detected objects in global coordinates
                det_boxes : numpy.ndarray
                    The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
                det_class : numpy.ndarray
                    The object class for each predicted bounding box, shape (N, 1) corresponding to the above bounding box. 
                    0 for vehicle, 1 for pedestrian, 2 for cyclist.
                det_score : numpy.ndarray
                    The confidence score for each predicted bounding box, shape (N, 1) corresponding to the above bounding box.
        """
        # Initialize the model
        config_file = '/data/UCR_student/sds/code/lab-1-perception-bundle-adjuster/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
        checkpoint_file = '/data/UCR_student/sds/code/lab-1-perception-bundle-adjuster/model/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
        # model_inferencer = MultiModalityDet3DInferencer(model=config_file,weights=checkpoint_file, device='cuda:0')
        inferencer = LidarDet3DInferencer(config_file, checkpoint_file, device='cuda:0')
        # model = init_model(config_file, checkpoint_file, device='cuda:0')
        print("Model initialized")

        # Prepare the input data for the model
        images = []
        lidar_data = None
        for sensor_id, (frame_id, data) in sensor_data.items():
            if sensor_id in ['Left', 'Right']:
                images.append(data[:, :, :3])  # Use only RGB channels
            elif sensor_id == 'LIDAR':
                lidar_data = data

        # Perform inference
        # results = inference_multi_modality_detector(model, images, lidar_data)
        results = inferencer({"points":lidar_data})

        # Process the results
        det_boxes = []
        det_class = []
        det_score = []

        preds = results['predictions']

        # import pdb; pdb.set_trace()        
        for result in preds:
            bbox = LiDARInstance3DBoxes(result['bboxes_3d'])
            det_boxes.append(np.array(bbox.corners).astype(int))
            for label in result['labels_3d']:
                det_class.append(label)
            for score in result['scores_3d']:
                det_score.append(score)

        det_boxes = np.array(det_boxes).reshape(-1, 8, 3)
        det_class = np.array(det_class).reshape(-1, 1)
        det_score = np.array(det_score).reshape(-1, 1)

        import pdb; pdb.set_trace()

        return {
            'det_boxes': det_boxes,
            'det_class': det_class,
            'det_score': det_score
        }

    