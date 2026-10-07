from sklearn.linear_model import LinearRegression

# Training data
X = [[1], [2], [3], [4], [5]]
y = [35, 45, 55, 65, 75]

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Take study hours from the user
hours = float(input("Enter your study hours: "))

# Predict marks
prediction = model.predict([[hours]])

print("Predicted marks:", prediction[0])