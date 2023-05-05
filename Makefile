export MAKEFLAGS=-j1

get-reference-repo-dryRun:
	python3 ../../scripts/reference-demo-clone-from-github.py --titleName "" --fileName README.md --dryRun
get-reference-repo:
	python3 ../../scripts/reference-demo-clone-from-github.py --titleName "" --fileName README.md

analysis-classifaction-models-info:
	python src/analysis_scenario_models.py --nnType 'classifaction'
analysis-object_detection-models-info:
	python src/analysis_scenario_models.py --nnType 'object_detection'
analysis-nlp:
	python src/analysis_scenario_models.py --nnType 'nlp'
analysis-face_recognition-models-info:
	python src/analysis_scenario_models.py --nnType 'face_recognition'
analysis-super_resolution-models-info:
	python src/analysis_scenario_models.py --nnType 'super_resolution'
analysis-pose_estimation-models-info:
	python src/analysis_scenario_models.py --nnType 'pose_estimation'
analysis-recommeder_systems-models-info:
	python src/analysis_scenario_models.py --nnType 'recommeder_systems'
analysis-cv_ocr-models-info:
	python src/analysis_scenario_models.py --nnType 'cv_ocr'
analysis-evolution-all-models-info:
	python src/analysis_scenario_models.py --nnType 'all'

analysis-demo-info:
	python src/demo_Res2Net.py

analysis-torch-models-info:
	python src/analysis_torch_models.py

regression-test:analysis-demo-info analysis-classifaction-models-info analysis-torch-models-info

clean:
	-rm -rf ./output/*
