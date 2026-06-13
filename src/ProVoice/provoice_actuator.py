import datetime

class ProVoiceActuator:
    def __init__(self):
        self.last_action = None
        self.last_message = ""
        self._last_key = None

    def execute(self, action: dict):
        self.last_action = action
        if not action or 'action' not in action:
            return  

        act_type = action.get('action')
        level = action.get('level', 'low')
        message = action.get('message', '')
        loa = action.get('LoA', None)   
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        loa_desc = ""
        if loa is not None:
            loa_map = {
                0: "none",
                1: "suggest",
                2: "ask_approval",
                3: "auto_with_veto",
                4: "auto"
            }
            loa_desc = f"（Present LoA={loa}: {loa_map.get(loa, '')}）"

      
        # Only log to the console when the decision actually changes — otherwise
        # this floods stdout with identical lines at the sample rate.
        key = (act_type, level, message, loa)
        changed = key != self._last_key
        self._last_key = key

        if act_type in ["none", None]:
            self.last_message = f"{timestamp} Normal：{message} {loa_desc}"
        elif act_type == "suggest":
            self.last_message = f"{timestamp} Advice：{message} {loa_desc}"
            changed and print(f"[Actuator] Suggestion: {message}")
        elif act_type == "ask_approval":
            self.last_message = f"{timestamp} Need approval：{message} {loa_desc}"
            changed and print(f"[Actuator] Ask for Approval: {message}")
        elif act_type == "auto_with_veto":
            self.last_message = f"{timestamp} Auto with veto：{message} {loa_desc}"
            changed and print(f"[Actuator] Auto Action (with veto): {message}")
        elif act_type == "auto":
            self.last_message = f"{timestamp} Auto：{message} {loa_desc}"
            changed and print(f"[Actuator] Fully Auto Action: {message}")
        elif act_type == "alert":
            if level == "high":
                self.last_message = f"{timestamp} Warning：{message} {loa_desc}"
                changed and print(f"[Actuator] Warning: {message}")
            elif level == "medium":
                self.last_message = f"{timestamp} Advice：{message} {loa_desc}"
                changed and print(f"[Actuator] Notice: {message}")
            else:
                self.last_message = f"{timestamp} Message：{message} {loa_desc}"
                changed and print(f"[Actuator] Message: {message}")
        else:
            self.last_message = f"{timestamp} Action：{message} {loa_desc}"
            changed and print(f"[Actuator] Action: {message}")