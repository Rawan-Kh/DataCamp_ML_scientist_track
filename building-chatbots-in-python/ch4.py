# Define the INIT state
INIT =0

# Define the CHOOSE_COFFEE state
CHOOSE_COFFEE =1

# Define the ORDERED state
ORDERED=2

# Define the policy rules
policy = {
    (INIT, "order"): (CHOOSE_COFFEE, "ok, Colombian or Kenyan?"),
    (INIT, "none"): (INIT, "I'm sorry - I'm not sure how to help you"),
    (CHOOSE_COFFEE, "specify_coffee"): (ORDERED, "perfect, the beans are on their way!"),
    (CHOOSE_COFFEE, "none"): (CHOOSE_COFFEE, "I'm sorry - would you like Colombian or Kenyan?"),
}

# Create the list of messages
messages = [
    "I'd like to become a professional dancer",
    "well then I'd like to order some coffee",
    "my favourite animal is a zebra",
    "kenyan"
]

# Call send_message() for each message
state = INIT
for message in messages:    
    state = send_message(policy, state, message)
 
# You just built your first chatbot with a state machine
-------------------

# Define the states
INIT=0 
CHOOSE_COFFEE=1
ORDERED=2

# Define the policy rules dictionary
policy_rules = {
    (INIT, "ask_explanation"): (INIT, "I'm a bot to help you order coffee beans"),
    (INIT, "order"): (CHOOSE_COFFEE, "ok, Colombian or Kenyan?"),
    (CHOOSE_COFFEE, "specify_coffee"): (ORDERED, "perfect, the beans are on their way!"),
    (CHOOSE_COFFEE, "ask_explanation"): (CHOOSE_COFFEE, "We have two kinds of coffee beans - the Kenyan ones make a slightly sweeter coffee, and cost $6. The Brazilian beans make a nutty coffee and cost $5.")    
}

# Define send_messages()
def send_messages(messages):
    state = INIT
    for msg in messages:
        state = send_message(state,msg)

# Send the messages
send_messages([
    "what can you do for me?",
    "well then I'd like to order some coffee",
    "what do you mean by that?",
    "kenyan"
])

# Your bot can now modify its answers by considering context
---------------
# Define respond()
def respond(message,params, prev_suggestions, excluded):
    # Interpret the message
    parse_data = interpret(message)
    # Extract the intent
    intent = parse_data["intent"]["name"]
    # Extract the entities
    entities = parse_data["entities"]
    # Add the suggestion to the excluded list if intent is "deny"
    if intent == "deny":
        excluded.extend(prev_suggestions)
    # Fill the dictionary with entities	
    for ent in entities:
        params[ent["entity"]] = str(ent["value"])
    # Find matching hotels
    results = [
        r 
        for r in find_hotels(params, excluded) 
        if r[0] not in excluded
    ]
    # Extract the suggestions
    names = [r[0] for r in results]
    n = min(len(results), 3)
    suggestions = names[:2]
    return responses[n].format(*names), params, suggestions, excluded

# Initialize the empty dictionary and lists
params, suggestions, excluded = {}, [], []

# Send the messages
for message in ["I want a mid range hotel", "no that doesn't work for me"]:
    print("USER: {}".format(message))
    response, params, suggestions, excluded = respond(message, params, suggestions, excluded)
    print("BOT: {}".format(response))

# Your bot can now handle negative feedback gracefully.
---------------------    

# Define policy()
def policy(intent):
    # Return "do_pending" if the intent is "affirm"
    if intent == "affirm":
        return "do_pending", None
    # Return "Ok" if the intent is "deny"
    if intent == "deny":
        return "Ok", None
    if intent == "order":
        return "Unfortunately, the Kenyan coffee is currently out of stock, would you like to order the Brazilian beans?", "Alright, I've ordered that for you!"

#  With a policy() function defined, you can now incorporate it into a send_message() function.      
----------------

# Define send_message()
def send_message(pending , message):
    print("USER : {}".format(message))
    action, pending_action = policy(interpret(message))
    if action == "do_pending" and pending is not None:
        print("BOT : {}".format(pending))
    else:
        print("BOT : {}".format(action))
    return pending_action
    
# Define send_messages()
def send_messages(messages):
    pending = None
    for msg in messages:
        pending = send_message(pending, msg)

# Send the messages
send_messages([
    "I'd like to order some coffee",
    "ok yes please"
])
# Your bot can now follow up on its own suggestions!
-------------

