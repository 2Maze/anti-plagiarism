from typing import List

def shift(steps_number: int, values: List[int]) -> List[int]:
    return [v + steps_number for v in values]

  
def mult(first: int, second: int) -> int:
    return first * second
