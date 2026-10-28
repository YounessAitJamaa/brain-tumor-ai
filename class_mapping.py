
# Fixed class order for the classification model
CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# Convert class names into numbers
CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

# Convert numbers back into class names
ID_TO_CLASS = {
    index: name
    for index, name in enumerate(CLASS_NAMES)
}
