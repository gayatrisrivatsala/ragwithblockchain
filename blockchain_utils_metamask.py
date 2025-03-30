# blockchain_utils_metamask.py
import hashlib
import json
import os
import time
import streamlit as st
from web3 import Web3

class BlockchainManagerMetaMask:
    def __init__(self, 
                 contract_address=None,
                 network_id=None):
        """
        Initialize blockchain connection for MetaMask integration.
        
        Args:
            contract_address: Address of the deployed RAG Document Verifier contract
            network_id: Network ID from MetaMask (e.g., '1' for Ethereum Mainnet)
        """
        self.contract_address = contract_address
        self.network_id = network_id
        
        # MetaMask connection status
        self.is_connected = False
        self.user_address = None
        
        # Contract ABI - Load from file or define here
        self.abi = [
            {
                "inputs": [],
                "stateMutability": "nonpayable",
                "type": "constructor"
            },
            {
                "anonymous": False,
                "inputs": [
                    {
                        "indexed": True,
                        "internalType": "address",
                        "name": "user",
                        "type": "address"
                    },
                    {
                        "indexed": True,
                        "internalType": "string",
                        "name": "documentId",
                        "type": "string"
                    },
                    {
                        "indexed": False,
                        "internalType": "string",
                        "name": "documentHash",
                        "type": "string"
                    },
                    {
                        "indexed": False,
                        "internalType": "uint256",
                        "name": "timestamp",
                        "type": "uint256"
                    }
                ],
                "name": "DocumentVerified",
                "type": "event"
            },
            {
                "anonymous": False,
                "inputs": [
                    {
                        "indexed": True,
                        "internalType": "address",
                        "name": "user",
                        "type": "address"
                    },
                    {
                        "indexed": True,
                        "internalType": "string",
                        "name": "queryId",
                        "type": "string"
                    },
                    {
                        "indexed": False,
                        "internalType": "string",
                        "name": "queryHash",
                        "type": "string"
                    },
                    {
                        "indexed": False,
                        "internalType": "uint256",
                        "name": "timestamp",
                        "type": "uint256"
                    }
                ],
                "name": "QueryLogged",
                "type": "event"
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "",
                        "type": "string"
                    }
                ],
                "name": "documentHashes",
                "outputs": [
                    {
                        "internalType": "string",
                        "name": "",
                        "type": "string"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "documentId",
                        "type": "string"
                    }
                ],
                "name": "getDocumentHash",
                "outputs": [
                    {
                        "internalType": "string",
                        "name": "",
                        "type": "string"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "queryId",
                        "type": "string"
                    }
                ],
                "name": "getQueryInfo",
                "outputs": [
                    {
                        "internalType": "string",
                        "name": "",
                        "type": "string"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "queryId",
                        "type": "string"
                    },
                    {
                        "internalType": "string",
                        "name": "queryHash",
                        "type": "string"
                    }
                ],
                "name": "logQuery",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "",
                        "type": "string"
                    }
                ],
                "name": "queryHashes",
                "outputs": [
                    {
                        "internalType": "string",
                        "name": "",
                        "type": "string"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "documentId",
                        "type": "string"
                    },
                    {
                        "internalType": "string",
                        "name": "documentHash",
                        "type": "string"
                    }
                ],
                "name": "verifyDocument",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
        # Flag to track if we're in a browser environment
        self.is_browser_env = self._check_browser_environment()
        
        # Generate JavaScript for web3 interaction with MetaMask
        self.web3_js = self._generate_web3_js()
    
    def _check_browser_environment(self):
        """Check if we're running in a browser environment."""
        # This is a simplified check - in a real app, you'd detect this differently
        return True
    
    def _generate_web3_js(self):
        """Generate JavaScript code for MetaMask interaction."""
        js_code = """
        <script>
            async function verifyDocumentWithMetaMask(documentId, documentHash) {
                if (typeof window.ethereum === 'undefined') {
                    return {
                        status: false,
                        error: "MetaMask is not installed. Please install MetaMask and try again."
                    };
                }

                try {
                    // Request account access if needed
                    const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
                    const account = accounts[0];
                    
                    // Create web3 instance
                    const web3 = new Web3(window.ethereum);
                    
                    // Create contract instance
                    const contractABI = CONTRACT_ABI_PLACEHOLDER;
                    const contractAddress = 'CONTRACT_ADDRESS_PLACEHOLDER';
                    const contract = new web3.eth.Contract(contractABI, contractAddress);
                    
                    // Prepare and send transaction
                    const tx = await contract.methods.verifyDocument(documentId, documentHash).send({
                        from: account
                    });
                    
                    return {
                        status: true,
                        tx_hash: tx.transactionHash,
                        document_id: documentId,
                        document_hash: documentHash,
                        block_number: tx.blockNumber,
                        status_code: tx.status ? 1 : 0
                    };
                } catch (error) {
                    console.error("Error verifying document:", error);
                    return {
                        status: false,
                        error: error.message
                    };
                }
            }
            
            async function logQueryWithMetaMask(queryId, queryHash) {
                if (typeof window.ethereum === 'undefined') {
                    return {
                        status: false,
                        error: "MetaMask is not installed. Please install MetaMask and try again."
                    };
                }

                try {
                    // Request account access if needed
                    const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
                    const account = accounts[0];
                    
                    // Create web3 instance
                    const web3 = new Web3(window.ethereum);
                    
                    // Create contract instance
                    const contractABI = CONTRACT_ABI_PLACEHOLDER;
                    const contractAddress = 'CONTRACT_ADDRESS_PLACEHOLDER';
                    const contract = new web3.eth.Contract(contractABI, contractAddress);
                    
                    // Prepare and send transaction
                    const tx = await contract.methods.logQuery(queryId, queryHash).send({
                        from: account
                    });
                    
                    return {
                        status: true,
                        tx_hash: tx.transactionHash,
                        query_id: queryId,
                        query_hash: queryHash,
                        block_number: tx.blockNumber,
                        status_code: tx.status ? 1 : 0
                    };
                } catch (error) {
                    console.error("Error logging query:", error);
                    return {
                        status: false,
                        error: error.message
                    };
                }
            }
            
            async function checkDocumentVerifiedWithMetaMask(documentId) {
                if (typeof window.ethereum === 'undefined') {
                    return {
                        status: false,
                        error: "MetaMask is not installed. Please install MetaMask and try again."
                    };
                }

                try {
                    // Create web3 instance
                    const web3 = new Web3(window.ethereum);
                    
                    // Create contract instance
                    const contractABI = CONTRACT_ABI_PLACEHOLDER;
                    const contractAddress = 'CONTRACT_ADDRESS_PLACEHOLDER';
                    const contract = new web3.eth.Contract(contractABI, contractAddress);
                    
                    // Call the contract
                    const result = await contract.methods.getDocumentHash(documentId).call();
                    
                    return {
                        status: true,
                        document_id: documentId,
                        document_hash: result,
                        is_verified: result !== ""
                    };
                } catch (error) {
                    console.error("Error checking document:", error);
                    return {
                        status: false,
                        error: error.message
                    };
                }
            }
        </script>
        """
        
        # Replace placeholders with actual values
        js_code = js_code.replace('CONTRACT_ABI_PLACEHOLDER', json.dumps(self.abi))
        if self.contract_address:
            js_code = js_code.replace('CONTRACT_ADDRESS_PLACEHOLDER', self.contract_address)
            
        return js_code
    
    def update_connection(self, is_connected, user_address=None, network_id=None):
        """Update the connection status with MetaMask info."""
        self.is_connected = is_connected
        self.user_address = user_address
        if network_id:
            self.network_id = network_id
    
    def compute_file_hash(self, file_path):
        """
        Compute the SHA-256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Hexadecimal hash of the file
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read and update hash in chunks for memory efficiency
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
                
        return sha256_hash.hexdigest()
    
    def verify_document(self, document_id, file_path):
        """
        Verify a document by storing its hash on the blockchain using MetaMask.
        
        In a real Streamlit app, this would generate HTML/JS to call MetaMask.
        For this example, we'll use placeholders.
        
        Args:
            document_id: Unique identifier for the document
            file_path: Path to the document file
            
        Returns:
            dict: Transaction status info
        """
        if not self.is_connected:
            return {
                "status": False,
                "error": "MetaMask is not connected. Please connect your wallet first."
            }
            
        if not self.contract_address:
            return {
                "status": False,
                "error": "Contract address is not set."
            }
        
        try:
            # Compute document hash
            document_hash = self.compute_file_hash(file_path)
            
            # In a real app, we would use JavaScript to call MetaMask
            # For this example, we'll return a simulated successful response
            
            # Display MetaMask interaction info
            st.info("Please confirm the transaction in your MetaMask wallet...")
            
            # Simulate transaction process time
            with st.spinner("Waiting for transaction confirmation..."):
                time.sleep(3)  # Simulate transaction time
            
            # Simulate successful transaction
            tx_hash = "0x" + "".join([format(i, "02x") for i in os.urandom(32)])
            block_number = 12345678
            
            return {
                "status": True,
                "tx_hash": tx_hash,
                "document_id": document_id,
                "document_hash": document_hash,
                "block_number": block_number,
                "status": 1  # Success
            }
            
        except Exception as e:
            st.error(f"Error during verification: {str(e)}")
            return {
                "status": False,
                "error": str(e)
            }
    
    def check_document_verified(self, document_id):
        """
        Check if a document has already been verified on the blockchain.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            bool: True if document is verified, False otherwise
        """
        if not self.is_connected:
            st.warning("MetaMask not connected. Cannot check document verification status.")
            return False
            
        if not self.contract_address:
            st.warning("Contract address not set. Cannot check document verification status.")
            return False
        
        # In a real app, we would use JavaScript to call the smart contract
        # For this example, we'll return a simulated response
        return False
    
    def log_query(self, query_text, answer_text):
        """
        Log a query and its answer on the blockchain.
        
        Args:
            query_text: The user's query
            answer_text: The system's answer
            
        Returns:
            dict: Transaction status info
        """
        if not self.is_connected:
            return {
                "status": False,
                "error": "MetaMask is not connected. Please connect your wallet first."
            }
            
        if not self.contract_address:
            return {
                "status": False,
                "error": "Contract address is not set."
            }
        
        try:
            # Create a unique query ID using timestamp
            query_id = f"query_{int(time.time())}"
            
            # Create a JSON object with query and answer
            query_data = {
                "query": query_text,
                "answer": answer_text,
                "timestamp": int(time.time())
            }
            
            # Hash the query data
            query_hash = hashlib.sha256(json.dumps(query_data).encode()).hexdigest()
            
            # Display MetaMask interaction info
            st.info("Please confirm the transaction in your MetaMask wallet...")
            
            # Simulate transaction process time
            with st.spinner("Waiting for transaction confirmation..."):
                time.sleep(3)  # Simulate transaction time
            
            # Simulate successful transaction
            tx_hash = "0x" + "".join([format(i, "02x") for i in os.urandom(32)])
            block_number = 12345678
            
            return {
                "status": True,
                "tx_hash": tx_hash,
                "query_id": query_id,
                "query_hash": query_hash,
                "block_number": block_number,
                "status": 1  # Success
            }
            
        except Exception as e:
            st.error(f"Error during query logging: {str(e)}")
            return {
                "status": False,
                "error": str(e)
            }
    
    def get_metamask_js(self):
        """Return the JavaScript code for MetaMask interaction."""
        return self.web3_js