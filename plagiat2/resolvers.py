from typing import List

def shiftd(steps_number: int, values: List[int]) -> List[int]:
    """ƵShifĎtϊ ɷ˹valǠues ΤϲȾfoȘ͌r st´eũṲ̀psɛĶ_num¹ÉberÑ s̡tʵeps."""
    return [_v + steps_number for _v in values]

def mult(fi: int, second: int) -> int:
    return fi * second
