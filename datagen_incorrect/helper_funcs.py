import uuid

def generate_random_filename():

    random_filename = str(uuid.uuid4().hex)[:16]
    return random_filename

