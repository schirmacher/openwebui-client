#!/usr/bin/env python3
"""
Open WebUI API Client Library

This module provides a Python class `OpenWebUIClient` for interacting with the
Open WebUI REST API, and a helper class `OpenWebUIMessageBuilder` for
constructing valid message payloads, including for multimodal inputs.

File uploads are handled by the OpenWebUIClient, not the message builder,
as they are attached at the top level of an API request.
"""

import requests
import json
import base64
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from contextlib import contextmanager


class OpenWebUIMessageBuilder:
    """
    A helper class to construct the 'messages' payload for the Open WebUI API.

    This builder simplifies the creation of message lists, especially for
    multimodal chats involving text and images. Note that it does not handle
    file attachments, which are managed by the OpenWebUIClient at the request level.
    """

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initializes the message builder.

        Args:
            system_prompt: An optional initial system message to set the
                           behavior of the assistant.
        """
        self.messages: List[Dict[str, Any]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_user_message(self, text: str) -> 'OpenWebUIMessageBuilder':
        """
        Adds a standard text-only user message.

        Args:
            text: The text content of the user's message.

        Returns:
            The builder instance for chaining.
        """
        self.messages.append({"role": "user", "content": text})
        return self

    def add_user_message_with_images(
            self, text: str, image_paths: List[str]
    ) -> 'OpenWebUIMessageBuilder':
        """
        Adds a multimodal user message containing text and one or more images.

        Args:
            text: The text prompt accompanying the images.
            image_paths: A list of local file paths to the images.

        Returns:
            The builder instance for chaining.

        Raises:
            FileNotFoundError: If any image path is invalid.
        """
        content_parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]

        for image_path in image_paths:
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found at path: {image_path}")

            # Guess MIME type and encode image to Base64
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type or not mime_type.startswith('image'):
                raise ValueError(f"Could not determine a valid image type for {image_path}")
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            image_url = {"url": f"data:{mime_type};base64,{encoded_string}"}
            content_parts.append({"type": "image_url", "image_url": image_url})

        self.messages.append({"role": "user", "content": content_parts})
        return self

    def add_assistant_message(self, text: str) -> 'OpenWebUIMessageBuilder':
        """
        Adds a message from the assistant, useful for few-shot prompting.

        Args:
            text: The text content of the assistant's message.

        Returns:
            The builder instance for chaining.
        """
        self.messages.append({"role": "assistant", "content": text})
        return self

    def build(self) -> List[Dict[str, Any]]:
        """
        Returns the constructed list of messages.

        Returns:
            The final list of message dictionaries for the API request.
        """
        return self.messages


class OpenWebUIClient:
    """
    A client for interacting with the Open WebUI REST API.
    """

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })

    def get_models(self) -> List[Dict[str, Any]]:
        """Retrieves a list of available models from the API."""
        url = f"{self.base_url}/api/models"
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get('data', [])

    def upload_file(self, file_path: str) -> Dict[str, Any]:
        """
        Uploads a file to be referenced in chat completions.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found at path: {file_path}")

        url = f"{self.base_url}/api/v1/files/"

        headers = {
            "Authorization": self.session.headers["Authorization"],
            "Accept": "application/json"
        }

        with open(path, "rb") as f:
            files_payload = {'file': f}
            response = requests.post(url, headers=headers, files=files_payload)

        response.raise_for_status()
        return response.json()

    def delete_file(self, file_id: str) -> None:
        """
        Deletes a previously uploaded file from the server.
        """
        url = f"{self.base_url}/api/v1/files/{file_id}"
        response = self.session.delete(url)
        response.raise_for_status()

    @contextmanager
    def upload_and_manage_files(self, file_paths: List[str]) -> Iterator[List[Dict[str, Any]]]:
        """
        A context manager to upload files and ensure they are deleted afterward.
        """
        uploaded_file_data = []
        try:
            for path in file_paths:
                uploaded_file_data.append(self.upload_file(path))
            yield uploaded_file_data
        finally:
            if not uploaded_file_data:
                return

            print("\nCleaning up temporary files from server...")
            for file_data in uploaded_file_data:
                file_id = file_data.get('id')
                if file_id:
                    try:
                        self.delete_file(file_id)
                        # Use 'filename' as that is what the API returns
                        filename = file_data.get('filename', 'Unknown Filename')
                        print(f"  - Deleted: {filename} (ID: {file_id})")
                    except requests.exceptions.RequestException as e:
                        print(f"  - WARNING: Failed to delete file {file_id}: {e}")

    def chat_completion(self, model: str, messages: List[Dict[str, Any]],
                        temperature: float = 0.7, max_tokens: Optional[int] = None,
                        response_format: Optional[Dict[str, Any]] = None, # <-- ADDED PARAMETER
                        uploaded_files: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Sends a request for a standard chat completion, with optional file attachments.
        """
        url = f"{self.base_url}/api/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": temperature, "stream": False}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if uploaded_files:
            payload['files'] = [{'id': f['id'], 'type': 'file'} for f in uploaded_files]

        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def stream_chat_completion(self, model: str, messages: List[Dict[str, Any]],
                               temperature: float = 0.7,
                               response_format: Optional[Dict[str, Any]] = None, # <-- ADDED PARAMETER
                               uploaded_files: Optional[List[Dict[str, Any]]] = None) -> Iterator[Dict[str, Any]]:
        """
        Sends a request for a streaming chat completion, with optional file attachments.
        """
        url = f"{self.base_url}/api/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": temperature, "stream": True}
        if response_format is not None:
            payload["response_format"] = response_format
        if uploaded_files:
            payload['files'] = [{'id': f['id'], 'type': 'file'} for f in uploaded_files]

        with self.session.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        print(f"\nWarning: Could not decode JSON chunk: {data_str}\n")
                        continue