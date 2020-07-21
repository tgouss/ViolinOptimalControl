def __add_to_nlp(self, param_name, param, duplicate_if_size_is_one):
    if isinstance(param, (list, tuple)):
        if len(param) != self.nb_phases:
            raise RuntimeError(
                param_name
                + " size("
                + str(len(param))
                + ") does not correspond to the number of phases("
                + str(self.nb_phases)
                + ")."
            )
        else:
            for i in range(self.nb_phases):
                self.nlp[i][param_name] = param[i]
    else:
        if self.nb_phases == 1:
            self.nlp[0][param_name] = param
