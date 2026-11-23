import requests
from openai import OpenAI

class LLM:
    def __init__(self, model, use_api=False, api_key=None, base_url=None):
        """
        Initialize the LLM instance.
        
        Args:
            model (str): The model name or path.
            use_api (bool): Whether to use an API endpoint or a local server.
            api_key (str): API key for the API endpoint (required if use_api=True).
            base_url (str): Base URL for the API endpoint (required if use_api=True).
        """
        self.model = model
        self.use_api = use_api
        self.api_key = api_key
        self.base_url = base_url
        self.default_params = {
            "temperature": 0.6,
            "max_tokens": 4096,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        # Validate API parameters if needed
        if self.use_api and (not self.api_key or not self.base_url):
            raise ValueError("API key and base URL are required for API usage")

    def execute_prompt(self, prompts, inputs=None, params=None, structured_response=False):
        """
        Query either a local LLM or an API endpoint.
        
        Args:
            prompt_template (str): The template for the prompt, with placeholders for inputs.
            inputs (dict, optional): A dictionary of inputs to substitute into the prompt template.
            params (dict, optional): Parameters for the LLM (overrides defaults).
            structured_response (bool, optional): Whether to return a structured response (e.g., JSON).
            
        Returns:
            str or dict: The response from the LLM.
        """
        # Merge provided params with defaults
        final_params = {**self.default_params, **(params or {})}

        # Manage prompts
        prompt_system = prompts['system']
        prompt_user = self._format_prompt(prompts['user'], inputs)

        try:
            # Query the LLM
            response = self._call(final_params, prompt_user, prompt_system)
            
            # Return structured response if requested
            if structured_response:
                return self._parse_structured_response(response)
            
            return response
        
        except Exception as e:
            print(f"Query error: {str(e)}")
            return None

    def _format_prompt(self, prompt_template, inputs=None):
        """
        Format the prompt by substituting placeholders with actual values.
        
        Args:
            prompt_template (str): The template for the prompt.
            inputs (dict, optional): A dictionary of inputs to substitute into the template.
            
        Returns:
            str: The formatted prompt.
        """
        # Default to an empty dictionary if no inputs are provided
        inputs = inputs or {}

        # Escape curly braces in input strings and substitute placeholders
        formatted_inputs = {
            key: value.replace("{", "{{").replace("}", "}}") if isinstance(value, str) else ""
            for key, value in inputs.items()
        }

        # Substitute placeholders in the template
        return prompt_template.format(**formatted_inputs)

    def _call(self, params, prompt_user, prompt_system="You are a helpful assistant."):
        """
        Query the LLM (either via API or local server).
        
        Args:
            prompt (str): The formatted prompt.
            params (dict): Parameters for the LLM.
            
        Returns:
            str: The response from the LLM.
        """
        if self.use_api:  # Use OpenAI client for API calls
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user}
                ],
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
                top_p=params["top_p"],
                frequency_penalty=params["frequency_penalty"],
                presence_penalty=params["presence_penalty"]
            )
            return response.choices[0].message.content
        
        else:  # Use local server
            response = requests.post(
                self.base_url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": prompt_system},
                        {"role": "user", "content": prompt_user}
                    ],
                    "temperature": params["temperature"],
                    "max_tokens": params["max_tokens"],
                    "top_p": params["top_p"],
                    "frequency_penalty": params["frequency_penalty"],
                    "presence_penalty": params["presence_penalty"]
                }
            )
            return response.json()["choices"][0]["message"]["content"]

    def _parse_structured_response(self, response):
        """
        Parse the response into a structured format (e.g., JSON).
        
        Args:
            response (str): The raw response from the LLM.
            
        Returns:
            dict: The parsed structured response.
        """
        try:
            # Attempt to parse the response as JSON
            return eval(response)  # Use `eval` cautiously; consider `json.loads` if the response is guaranteed JSON
        except Exception as e:
            print(f"Failed to parse structured response: {str(e)}")
            return {"error": "Failed to parse structured response", "raw_response": response}