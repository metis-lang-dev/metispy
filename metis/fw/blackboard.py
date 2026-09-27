"""The shared store the evaluator speaks through. A dict with explicit
presence checking — python introspection is the typing story for now
(the C++ port will make ports statically typed)."""


class Blackboard(dict):
    def require(self, names):
        missing = [n for n in names if n not in self]
        if missing:
            raise KeyError("blackboard missing inputs: "
                           + ", ".join(sorted(missing)))
