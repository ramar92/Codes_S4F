
def calculator(expression):
    allowed = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
    }

    operation, a, b = expression
    return allowed[operation](a, b)


def simple_agent(goal):
    print("Goal:", goal)

    # Demonstration of a predefined planning rule
    if goal == "calculate_total":
        plan = ("add", 1500, 2500)
        print("Plan: Use the addition tool")
        result = calculator(plan)
        return f"Total amount: ₹{result}"

    return "I need a suitable tool for this task."


print(simple_agent("calculate_total"))
