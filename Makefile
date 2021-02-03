clean-results:
	@echo "Cleaning model saves and plots..."
	@rm -rvf categorization/model_saves/nose/*
	@rm -rvf categorization/model_saves/face/*
	@rm -rvf categorization/model_saves/eyes/*
	@rm -rvf categorization/model_saves/skin/*
	@rm -rvf categorization/model_saves/mouth/*
	@rm -rvf categorization/model_saves/stacked/*
	@rm -rvf data/plots/*

clean-data:
	@echo "Removing data extracted from unparsed images..."
	@rm -rvf data/parsed/validation_sick/*
	@rm -rvf data/parsed/validation_healthy/*
	@rm -rvf data/parsed/sick/*
	@rm -rvf data/parsed/healthy/*
	@rm -rvf data/parsed/brightened/sick/*
	@rm -rvf data/parsed/brightened/healthy/*

create-data:
	@echo "Extracting data from images..."
# 	@python augment/face_org.py
	@python augment/alter_images.py

train-stacked:
	@echo "Training and cross-validating the stacking ensemble..."
	@python categorization/cross_val_stacked.py

train-individual:
	@echo "Training and cross-validating individual models..."
	@python categorization/cross_val_cnn.py