import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# داده‌های آموزشی
messages = [
    "Congratulations! You won $1000!",
    "You have been selected for a free prize!",
    "Claim your reward now!",
    "Hey, are you coming to class tomorrow?",
    "Don't forget to send me the homework.",
    "Can you call me when you get home?"
]

labels = [
    "spam",
    "spam",
    "spam",
    "not spam",
    "not spam",
    "not spam"
]

# ساخت DataFrame
data = pd.DataFrame({
    "message": messages,
    "label": labels
})

# تبدیل متن به عدد
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["message"])

# ساخت و آموزش مدل
model = MultinomialNB()
model.fit(X, data["label"])

# گرفتن پیام از کاربر
new_message = input("Enter a message: ")

# تبدیل پیام جدید به عدد
new_message_vector = vectorizer.transform([new_message])

# پیش‌بینی
prediction = model.predict(new_message_vector)

print("Prediction:", prediction[0])