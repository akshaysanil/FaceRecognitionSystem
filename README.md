# FaceRecognitionSystem


##the whole project Tree(for better understanding)
├── model
│   │   ├── 20180402-114759.pb
│   │   ├── model-20180402-114759.ckpt-275.data-00000-of-00001
│   │   ├── model-20180402-114759.ckpt-275.index
│   │   ├── model-20180402-114759.meta
│   │   └── new_classifier_jun13.pkl
│   ├── new_classifier_with160x160.pkl
│   ├── new_classifier_with_withNormalWhl160x160.pkl
│   ├── pretrained_checkpiont
│   │   └── 20170512-110547
│   │       ├── 20170512-110547.pb
│   │       ├── model-20170512-110547.ckpt-250000.data-00000-of-00001
│   │       ├── model-20170512-110547.ckpt-250000.index
│   │       └── model-20170512-110547.meta
│   ├── .project
│   ├── __pycache__
│   │   ├── facent_svm_rec_passing.cpython-38.pyc
│   │   └── __init__.cpython-38.pyc
│   ├── .pydevproject
│   ├── .pylintrc
│   ├── README.md
│   ├── requirements.txt
│   ├── src
│   │   ├── align
│   │   │   ├── align_dataset_mtcnn.py
│   │   │   ├── det1.npy
│   │   │   ├── det2.npy
│   │   │   ├── det3.npy
│   │   │   ├── detect_face.py
│   │   │   ├── __init__.py
│   │   │   └── __pycache__
│   │   │       └── detect_face.cpython-38.pyc
│   │   ├── calculate_filtering_metrics.py
│   │   ├── classifier.py
│   │   ├── compare.py
│   │   ├── decode_msceleb_dataset.py
│   │   ├── download_and_extract.py
│   │   ├── facenet.py
│   │   ├── freeze_graph.py
│   │   ├── generative
│   │   │   ├── calculate_attribute_vectors.py
│   │   │   ├── __init__.py
│   │   │   ├── models
│   │   │   │   ├── dfc_vae_large.py
│   │   │   │   ├── dfc_vae.py
│   │   │   │   ├── dfc_vae_resnet.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── vae_base.py
│   │   │   ├── modify_attribute.py
│   │   │   └── train_vae.py
│   │   ├── __init__.py
│   │   ├── lfw.py
│   │   ├── models
│   │   │   ├── dummy.py
│   │   │   ├── inception_resnet_v1.py
│   │   │   ├── inception_resnet_v2.py
│   │   │   ├── __init__.py
│   │   │   └── squeezenet.py
│   │   ├── __pycache__
│   │   │   └── facenet.cpython-38.pyc
│   │   ├── train_softmax.py
│   │   ├── train_tripletloss.py
│   │   └── validate_on_lfw.py
│   ├── test
│   │   ├── batch_norm_test.py
│   │   ├── center_loss_test.py
│   │   ├── restore_test.py
│   │   ├── train_test.py
│   │   └── triplet_loss_test.py
│   ├── tmp
│   │   ├── align_dataset.m
│   │   ├── align_dataset.py
│   │   ├── align_dlib.py
│   │   ├── cacd2000_split_identities.py
│   │   ├── dataset_read_speed.py
│   │   ├── deepdream.py
│   │   ├── detect_face_v1.m
│   │   ├── detect_face_v2.m
│   │   ├── download_vgg_face_dataset.py
│   │   ├── funnel_dataset.py
│   │   ├── __init__.py
│   │   ├── invariance_test.txt
│   │   ├── mnist_center_loss.py
│   │   ├── mnist_noise_labels.py
│   │   ├── mtcnn.py
│   │   ├── mtcnn_test_pnet_dbg.py
│   │   ├── mtcnn_test.py
│   │   ├── network.py
│   │   ├── nn2.py
│   │   ├── nn3.py
│   │   ├── nn4.py
│   │   ├── nn4_small2_v1.py
│   │   ├── pilatus800.jpg
│   │   ├── random_test.py
│   │   ├── rename_casia_directories.py
│   │   ├── seed_test.py
│   │   ├── select_triplets_test.py
│   │   ├── test1.py
│   │   ├── test_align.py
│   │   ├── test_invariance_on_lfw.py
│   │   ├── vggface16.py
│   │   ├── vggverydeep19.py
│   │   ├── visualize.py
│   │   ├── visualize_vggface.py
│   │   └── visualize_vgg_model.py
│   ├── .travis.yml
│   ├── util
│   │   └── plot_learning_curves.m
│   └── wetransfer_20170512-110547_2024-06-13_1220.zip
├── facenet_files
│   ├── facenet_svm_passing_withoutNPZ.py
│   ├── facent_svm_facepassing_backup.py
│   ├── facent_svm_facepassing_copy_V2.py
│   ├── facent_svm_rec_passing.py
│   └── __pycache__
│       └── facent_svm_rec_passing.cpython-38.pyc
├── facenet_models
│   ├── faces_embeddings_done_5members_V1.npz
│   └── faces_embeddings_done_5members_V2.npz
├── faces_embeddings_done_for_officeMysr.npz
├── .idea
│   ├── facenet_new.iml
│   ├── .gitignore
│   ├── inspectionProfiles
│   │   └── profiles_settings.xml
│   ├── misc.xml
│   ├── modules.xml
│   └── workspace.xml
├── importent_commands.odt
├── ip_cam1.py
├── lined_frame.png
├── live.png
├── marked_attendance
│   ├── 2024_06_16
│   │   ├── 2024_06_16_attendance_sheet.csv
│   │   ├── Ashlesha P D_2024_06_16_18:43:46.jpg
│   │   ├── Deepak A_2024_06_16_18:41:27.jpg
│   │   ├── Harshitha H_2024_06_16_18:42:22.jpg
│   │   ├── Namratha S Gowda_2024_06_16_18:41:09.jpg
│   │   └── Pooja Aiyappa_2024_06_16_18:41:24.jpg
│   ├── 2024_06_18
│   │   └── 2024_06_18_attendance_sheet.csv
│   ├── 2024_06_19
│   │   ├── 2024_06_19_attendance_sheet.csv
│   │   ├── Deepak A_2024_06_19_20:37:00.jpg
│   │   ├── Harshitha H_2024_06_19_20:36:03.jpg
│   │   └── Harshitha H_2024_06_19_20:40:05.jpg
│   ├── 2024_06_20
│   │   ├── 2024_06_20_attendance_sheet.csv
│   │   ├── Abhishek P_2024_06_20_01:38:53.jpg
│   │   ├── Deepak A_2024_06_20_01:27:26.jpg
│   │   ├── Deepak A_2024_06_20_01:35:29.jpg
│   │   ├── Harsha K S_2024_06_20_01:35:45.jpg
│   │   ├── Harshitha H_2024_06_20_01:24:03.jpg
│   │   ├── Namratha S Gowda_2024_06_20_01:35:34.jpg
│   │   ├── Pooja Aiyappa_2024_06_20_01:27:11.jpg
│   │   └── Satish S_2024_06_20_01:35:30.jpg
│   ├── core.py
│   └── core_updatedAfterPobability.py
├── __pycache__
│   ├── facenet_with_webcam.cpython-38.pyc
│   └── facent_svm_facepassing.cpython-38.pyc
├── requeirment.txt
├── requirementV2.txt
├── result_datas
│   ├── achuChay_tested.mp4
│   ├── face_from_yolo.png
│   ├── face.png
│   ├── output_tested.mp4
│   ├── result2.mp4
│   ├── result_harshitha.mp4
│   ├── result_ipcamera.mp4
│   ├── result.mp4
│   ├── result_output1.mp4
│   ├── result.png
│   ├── result_roi.jpeg
│   ├── result_roi.mp4
│   ├── result_roi.png
│   ├── result_testing1.mp4
│   └── tested.mp4
├── result_datastested.mp4
├── result_harshitha.mp4
├── rtsp.py
├── supervision
│   ├── annotators
│   │   ├── base.py
│   │   ├── core copy.py
│   │   ├── core.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── base.cpython-38.pyc
│   │   │   ├── core.cpython-38.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── utils.cpython-38.pyc
│   │   └── utils.py
│   ├── assets
│   │   ├── downloader.py
│   │   ├── __init__.py
│   │   ├── list.py
│   │   └── __pycache__
│   │       ├── downloader.cpython-38.pyc
│   │       ├── __init__.cpython-38.pyc
│   │       └── list.cpython-38.pyc
│   ├── classification
│   │   ├── core.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── core.cpython-38.pyc
│   │       └── __init__.cpython-38.pyc
│   ├── config.py
│   ├── dataset
│   │   ├── core.py
│   │   ├── formats
│   │   │   ├── coco.py
│   │   │   ├── __init__.py
│   │   │   ├── pascal_voc.py
│   │   │   ├── __pycache__
│   │   │   │   ├── coco.cpython-38.pyc
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── pascal_voc.cpython-38.pyc
│   │   │   │   └── yolo.cpython-38.pyc
│   │   │   └── yolo.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── core.cpython-38.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── utils.cpython-38.pyc
│   │   └── utils.py
│   ├── detection
│   │   ├── annotate.py
│   │   ├── core.py
│   │   ├── __init__.py
│   │   ├── line_zone.py
│   │   ├── lmm.py
│   │   ├── overlap_filter.py
│   │   ├── __pycache__
│   │   │   ├── annotate.cpython-38.pyc
│   │   │   ├── core.cpython-38.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── line_zone.cpython-38.pyc
│   │   │   ├── lmm.cpython-38.pyc
│   │   │   ├── overlap_filter.cpython-38.pyc
│   │   │   └── utils.cpython-38.pyc
│   │   ├── tools
│   │   │   ├── csv_sink.py
│   │   │   ├── inference_slicer.py
│   │   │   ├── __init__.py
│   │   │   ├── json_sink.py
│   │   │   ├── polygon_zone.py
│   │   │   ├── __pycache__
│   │   │   │   ├── csv_sink.cpython-38.pyc
│   │   │   │   ├── inference_slicer.cpython-38.pyc
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── json_sink.cpython-38.pyc
│   │   │   │   ├── polygon_zone.cpython-38.pyc
│   │   │   │   └── smoother.cpython-38.pyc
│   │   │   └── smoother.py
│   │   └── utils.py
│   ├── draw
│   │   ├── color.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── color.cpython-38.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── utils.cpython-38.pyc
│   │   └── utils.py
│   ├── geometry
│   │   ├── core.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── core.cpython-38.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── utils.cpython-38.pyc
│   │   └── utils.py
│   ├── __init__.py
│   ├── keypoint
│   │   ├── annotators.py
│   │   ├── core.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── annotators.cpython-38.pyc
│   │   │   ├── core.cpython-38.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── skeletons.cpython-38.pyc
│   │   └── skeletons.py
│   ├── metrics
│   │   ├── detection.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── detection.cpython-38.pyc
│   │       └── __init__.cpython-38.pyc
│   ├── __pycache__
│   │   ├── config.cpython-38.pyc
│   │   └── __init__.cpython-38.pyc
│   ├── tracker
│   │   ├── byte_tracker
│   │   │   ├── basetrack.py
│   │   │   ├── core.py
│   │   │   ├── __init__.py
│   │   │   ├── kalman_filter.py
│   │   │   ├── matching.py
│   │   │   └── __pycache__
│   │   │       ├── basetrack.cpython-38.pyc
│   │   │       ├── core.cpython-38.pyc
│   │   │       ├── __init__.cpython-38.pyc
│   │   │       ├── kalman_filter.cpython-38.pyc
│   │   │       └── matching.cpython-38.pyc
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       └── __init__.cpython-38.pyc
│   ├── utils
│   │   ├── conversion.py
│   │   ├── file.py
│   │   ├── image.py
│   │   ├── __init__.py
│   │   ├── internal.py
│   │   ├── iterables.py
│   │   ├── notebook.py
│   │   ├── __pycache__
│   │   │   ├── conversion.cpython-38.pyc
│   │   │   ├── file.cpython-38.pyc
│   │   │   ├── image.cpython-38.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── internal.cpython-38.pyc
│   │   │   ├── iterables.cpython-38.pyc
│   │   │   ├── notebook.cpython-38.pyc
│   │   │   └── video.cpython-38.pyc
│   │   └── video.py
│   └── validators
│       ├── __init__.py
│       └── __pycache__
│           └── __init__.cpython-38.pyc
├── svm_model_160x160_Office_mysr.pkl
├── svm_models
│   ├── svm_model_5memmbers_V1.pkl
│   ├── svm_model_5memmbers_V2.pkl
│   └── yoloCropSvmModel160x160.pkl
├── test_datas
│   ├── ipcamera.mp4
│   ├── output.mp4
│   ├── people _ walking _.mp4
│   ├── testingImg.jpeg
│   └── testing_video.mp4
├── training_gpu.py
├── training.py
├── wetransfer_svm_model_160x160_office_mysr-pkl_2024-06-13_0504.zip
├── working.ipynb
├── yoloFacenet.ipynb
├── yolo_models
│   └── yolov8n-face.pt
├── yolo_svm_with_line.py
├── yolo_with_facenet_main.py
├── yolo_with_facenet_rtsp.py
├── yolo_with_facenet_svm
│   ├── facenet_with_svm_main.py
│   ├── facenet_with_svm_webcam_default.py
│   ├── yolo_with_facenet_copy2.py
│   ├── yolo_with_facenet_copy.py
│   ├── yolo_with_facenet_for_img.py
│   ├── yolo_with_facenet_roi_copy.py
│   └── yolo_with_facenet_roi.py
└── yolo_with_facenet_withoutTracker.py

219 directories, 1904 files
