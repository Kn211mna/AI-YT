# Частотні таблиці (з пункту 2)
P_Yes = 10 / 14
P_No = 4 / 14

# Ймовірності для "Yes" та "No"
likelihoods = {
    'Outlook': {
        'Overcast': {'Yes': 4 / 10, 'No': 0 / 4},
        'Sunny': {'Yes': 3 / 10, 'No': 2 / 4},
        'Rain': {'Yes': 3 / 10, 'No': 2 / 4}
    },
    'Humidity': {
        'High': {'Yes': 3 / 9, 'No': 4 / 5},
        'Normal': {'Yes': 6 / 9, 'No': 1 / 5}
    },
    'Wind': {
        'Strong': {'Yes': 6 / 9, 'No': 2 / 5},
        'Weak': {'Yes': 3 / 9, 'No': 3 / 5}
    }
}

def predict_match(outlook, humidity, wind):
    # Ймовірностей для "Yes"
    P_Yes_given_conditions = (
        likelihoods['Outlook'][outlook]['Yes'] *
        likelihoods['Humidity'][humidity]['Yes'] *
        likelihoods['Wind'][wind]['Yes'] *
        P_Yes
    )

    # Ймовірностей для "No"
    P_No_given_conditions = (
        likelihoods['Outlook'][outlook]['No'] *
        likelihoods['Humidity'][humidity]['No'] *
        likelihoods['Wind'][wind]['No'] *
        P_No
    )

    # Нормалізація значень
    P_Total = P_Yes_given_conditions + P_No_given_conditions
    P_Yes_normalized = (P_Yes_given_conditions / P_Total * 100) if P_Total != 0 else 0
    P_No_normalized = (P_No_given_conditions / P_Total * 100) if P_Total != 0 else 0

    return P_Yes_normalized, P_No_normalized

# Вхідні дані
outlook = "Sunny"  # Перспектива
humidity = "Normal" # Вологість
wind = "Strong" # Вітер

# Результат прогнозування
P_Yes_result, P_No_result = predict_match(outlook, humidity, wind)
print(f"Ймовірність, що матч відбудеться (Yes): {P_Yes_result:.2f}%")
print(f"Ймовірність, що матч не відбудеться (No): {P_No_result:.2f}%")
