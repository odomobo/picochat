Unnecessary change in gpt:

        # Save hidden states before output projection (for conviction head)
        hidden_states = x

        # Conviction head operates on hidden states before output projection
        if self.config.use_conviction_head:
            conviction = self.conviction_head(hidden_states)

Review changes (which are really just to expose conviction head to outputs, and to train conviction head).

Next ticket will be to actually find a way to display conviction head when performing inference.