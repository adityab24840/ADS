class Employee:
    def __init__(self, name, age, salary):
        self.name = name
        self.age = age
        self.salary = salary
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, employee):
        self.root = self._insert(self.root, employee)

    def _insert(self, node, employee):
        if not node:
            return Employee(employee.name, employee.age, employee.salary)
        if employee.age < node.age:
            node.left = self._insert(node.left, employee)
        else:
            node.right = self._insert(node.right, employee)
        return node

    def search_employee(self, age):
        return self._search_employee(self.root, age)

    def _search_employee(self, node, age):
        if not node:
            return None
        if node.age == age:
            return node
        if age < node.age:
            return self._search_employee(node.left, age)
        return self._search_employee(node.right, age)

    def calculate_average_salary(self, node, min_age, max_age, total_salary, count):
        if not node:
            return
        if min_age <= node.age <= max_age:
            total_salary[0] += node.salary
            count[0] += 1
        if node.age > min_age:
            self.calculate_average_salary(node.left, min_age, max_age, total_salary, count)
        if node.age < max_age:
            self.calculate_average_salary(node.right, min_age, max_age, total_salary, count)

    def average_salary_between_ages(self, min_age, max_age):
        total_salary = [0]
        count = [0]
        self.calculate_average_salary(self.root, min_age, max_age, total_salary, count)
        if count[0] == 0:
            return 0
        return total_salary[0] / count[0]

# Example usage:
tree = BinaryTree()

# Insert N employee information
employees = [
    Employee("John", 25, 50000),
    Employee("Alice", 30, 60000),
    Employee("Bob", 22, 45000),
    Employee("Eve", 24, 52000),
    # Add more employees as needed
]

for employee in employees:
    tree.insert(employee)

# a. Search for an employee by age (e.g., age 25)
age_to_lookup = 25
found_employee = tree.search_employee(age_to_lookup)

if found_employee:
    print("Employee Found:", found_employee.name, "Age:", found_employee.age, "Salary:", found_employee.salary)
else:
    print("Employee not found")

# b. Calculate and display the average salary for employees between ages 20 and 25
min_age = 20
max_age = 25
average_salary = tree.average_salary_between_ages(min_age, max_age)
print("Average Salary for employees between 20 and 25:", average_salary)
