from itertools import combinations

from pyparsing import alphas, Combine, Literal, Optional, nums, Word


class UAIRead2(object):
    network_type = "MARKOV"

    def __init__(self, path=None, string=None):
        """
        Initialize an instance of UAI reader class

        Parameters
        ----------
        path : file or str
            Path of the file containing UAI information.

        string : str
            String containing UAI information.

        Examples
        --------

        Reference
        ---------
        http://graphmod.ics.uci.edu/uai08/FileFormat
        """
        if path:
            data = open(path)
            self.network = data.read()
        elif string:
            self.network = string
        else:
            raise ValueError("Must specify either path or string.")
        self.grammar = self.get_grammar()
        self.variables = self.get_variables()
        self.everything = self.get_everything()

    def get_everything(self):
        edges = []
        tables = []
        for function in range(0, self.no_functions):
            function_variables = self.grammar.parseString(self.network)[
                "fun_" + str(function)
                ]
            function_variables2 = function_variables
            if isinstance(function_variables, int):
                function_variables = [function_variables]
            if self.network_type == "MARKOV":
                function_variables = ["var_" + str(var) for var in function_variables]
                values = self.grammar.parseString(self.network)[
                    "fun_values_" + str(function)
                    ]
                tables.append((function_variables, list(values)))

            if isinstance(function_variables2, int):
                function_variables2 = [function_variables2]
            if self.network_type == "MARKOV":
                function_variables2 = ["var_" + str(var) for var in function_variables2]
                edges.extend(list(combinations(function_variables2, 2)))
        return tables, set(edges)

    def get_grammar(self):
        """
                Returns the grammar of the UAI file.
                """
        network_name = Word(alphas).setResultsName("network_name")
        no_variables = Word(nums).setResultsName("no_variables")
        grammar = network_name + no_variables
        self.no_variables = int(grammar.parseString(self.network)["no_variables"])
        domain_variables = (Word(nums) * self.no_variables).setResultsName(
            "domain_variables"
        )
        grammar += domain_variables
        no_functions = Word(nums).setResultsName("no_functions")
        grammar += no_functions
        self.no_functions = int(grammar.parseString(self.network)["no_functions"])
        integer = Word(nums).setParseAction(lambda t: int(t[0]))
        for function in range(0, self.no_functions):
            scope_grammar = Word(nums).setResultsName("fun_scope_" + str(function))
            grammar += scope_grammar
            function_scope = grammar.parseString(self.network)[
                "fun_scope_" + str(function)
                ]
            function_grammar = ((integer) * int(function_scope)).setResultsName(
                "fun_" + str(function)
            )
            grammar += function_grammar
        floatnumber = Combine(
            Word(nums) + Optional(Literal(".") + Optional(Word(nums)))
        )
        for function in range(0, self.no_functions):
            no_values_grammar = Word(nums).setResultsName(
                "fun_no_values_" + str(function)
            )
            grammar += no_values_grammar
            no_values = grammar.parseString(self.network)[
                "fun_no_values_" + str(function)
                ]
            values_grammar = ((floatnumber) * int(no_values)).setResultsName(
                "fun_values_" + str(function)
            )
            grammar += values_grammar
        return grammar

    def get_variables(self):
        variables = []
        for var in range(0, self.no_variables):
            var_name = "var_" + str(var)
            variables.append(var_name)
        return variables
