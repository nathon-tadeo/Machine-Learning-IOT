# C. Define a class "person"
class person:
    def __init__(self, name, age, height):
        self.name = name
        self.age = age
        self.height = height

# D. __repr__(self) that returns a string of this form: "Bob is 17 years old and 182 cm tall".
    def __repr__(self):
        return f"{self.name} is {self.age} years old and {self.height} cm tall"
