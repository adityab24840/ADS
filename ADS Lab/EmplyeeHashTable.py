class Employee:
    def __init__(self, name, age, salary):
        self.name = name
        self.age = age
        self.salary = salary

def insert_into_hash_table(employees):
    hash_table = {}
    for employee in employees:
        age = employee.age
        if age in hash_table:
            hash_table[age].append(employee)
        else:
            hash_table[age] = [employee]
    return hash_table

def search_employee(hash_table, age_to_lookup):
    if age_to_lookup in hash_table:
        return hash_table[age_to_lookup]
    else:
        return None

def average_salary_between_ages(hash_table, min_age, max_age):
    total_salary = 0
    count = 0
    for age in range(min_age, max_age + 1):
        if age in hash_table:
            for employee in hash_table[age]:
                total_salary += employee.salary
                count += 1
    if count == 0:
        return 0
    return total_salary / count

# Example usage:
employees = [
    Employee("John", 25, 50000),
    Employee("Alice", 30, 60000),
    Employee("Bob", 22, 45000),
    Employee("Eve", 24, 52000),
    # Add more employees as needed
]

hash_table = insert_into_hash_table(employees)

# a. Search for an employee by age (e.g., age 25)
age_to_lookup = 25  # Change this to the age you want to lookup
found_employees = search_employee(hash_table, age_to_lookup)

if found_employees:
    print(f"Employees with age {age_to_lookup}:")
    for employee in found_employees:
        print(f"Name: {employee.name}, Salary: {employee.salary}")
else:
    print(f"No employees found with age {age_to_lookup}")

# b. Calculate and display the average salary for employees between ages 22 and 25
min_age = 22
max_age = 25
average_salary = average_salary_between_ages(hash_table, min_age, max_age)
print(f"Average Salary for employees between {min_age} and {max_age}: {average_salary}")
