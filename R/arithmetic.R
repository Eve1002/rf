
#' Imitate randomforest
#'
#' @param train_data we split the original data randomly into 2 datasets, one is train data,usually 70% of the data
#' @param test_data we split the original data randomly into 2 datasets, another is test data,usually 30% of the data
#' @param target_variable the dependent variable that we would like to predict
#' @param num_trees number of the decision trees we create
#' @param label_mapping whether there are special algorithm for the target variable,eg for penguins:"1" = "Adelie", "2" = "Chinstrap", "3" = "Gentoo"
#'
#' @return prediction of the target variable based on the training dataset
#' @export
#'
#' @examples
#' library(palmerpenguins)
#' library(tidyverse)
#' library(dplyr)
#' library(stringr)
#' library(randomForest)
#' library(rpart)
#' data("penguins")
#' penguins <- na.omit(penguins)
#' penguins <- penguins %>% select(-year)
#' set.seed(123)
#' sample_indices <- sample(1:nrow(penguins), 0.7 * nrow(penguins))
#' train_data <- penguins[sample_indices, ]
#' test_data <- penguins[-sample_indices, ]
#' target_variable <- "species"
#' ensemble_pred <- ensemble_predict(train_data, test_data, target_variable)
ensemble_predict <- function(train_data, test_data, target_variable, num_trees = 100, label_mapping = NULL) {
  # Initialize an empty list to store individual decision trees
  tree_list <- list()

  # Build multiple decision trees
  for (i in 1:num_trees) {
    # Create a bootstrap sample (sampling with replacement)
    bootstrap_sample <- train_data[sample(1:nrow(train_data), replace = TRUE), ]

    # Build a decision tree using rpart
    formula <- as.formula(paste(target_variable, "~ ."))
    tree <- rpart(formula, data = bootstrap_sample, method = "class")

    # Add the tree to the list
    tree_list[[i]] <- tree
  }

  # Make predictions using the ensemble of decision trees
  predictions <- lapply(tree_list, function(tree) predict(tree, newdata = test_data, type = "class"))

  # Combine predictions (majority vote)
  ensemble_pred <- do.call(cbind, predictions)
  # If label mapping is not provided, create a default one for species
  if (is.null(label_mapping)) {
    label_mapping <- c("1" = "Adelie", "2" = "Chinstrap", "3" = "Gentoo")
  }

  # Map numerical labels to character labels
  final_pred <- apply(ensemble_pred, 1, function(row) {
    majority_vote <- names(sort(table(row), decreasing = TRUE)[1])
    return(label_mapping[majority_vote])
  })

  return(final_pred)
}




#' use cross-validation to tune hyperparameter(complexity parameter)
#'
#' @param train_data we split the original data randomly into 2 datasets, one is train data,usually 70% of the data
#' @param target the dependent variable that we would like to predict
#'
#' @return contains information about the cross-validated tuning process and the best-tuned random forest model
#' @export
#'
#' @examples
#' library(palmerpenguins)
#' library(tidyverse)
#' library(dplyr)
#' library(stringr)
#' library(randomForest)
#' library(rpart)
#' library(caret)
#' data("penguins")
#' penguins <- na.omit(penguins)
#' penguins <- penguins %>% select(-year)
#' set.seed(123)
#' sample_indices <- sample(1:nrow(penguins), 0.7 * nrow(penguins))
#' train_data <- penguins[sample_indices, ]
#' rf_cv <- rf_cv_function(train_data, 'species')
rf_cv_function <- function(train_data, target) {
  # Define the tuning grid which default is 0.01
  tuning_grid <- expand.grid(
    cp = seq(0.01, 0.1, by = 0.01)  # Adjust the range of cp as needed
  )

  # Define the train control
  ctrl <- trainControl(
    method = "cv",
    number = 5,  # Number of folds for cross-validation
    verboseIter = TRUE
  )

  # Cross-validation using train function
  rf_cv <- train(
    reformulate(names(train_data) %>% setdiff(target), response = target),
    data = train_data,
    method = "rpart",
    tuneGrid = tuning_grid,
    trControl = ctrl
  )

  # Print the best model
  print(rf_cv)

  # Return the trained model
  return(rf_cv)
}


