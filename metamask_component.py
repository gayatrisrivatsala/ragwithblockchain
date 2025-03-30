# metamask_component.py
import os
import json
import streamlit as st
import streamlit.components.v1 as components

# Define the MetaMask component HTML
def get_metamask_component_html():
    """
    Creates an HTML component that can interact with MetaMask browser extension.
    """
    return """
    <script src="https://cdnjs.cloudflare.com/ajax/libs/web3/1.8.0/web3.min.js"></script>
    <div id="metamask-connector">
        <div id="metamask-status">
            <p id="connection-status">MetaMask Status: Not Connected</p>
            <p id="wallet-address">Wallet: Not Connected</p>
            <p id="network-name">Network: Unknown</p>
        </div>
        <button id="connect-button" class="metamask-button">Connect MetaMask</button>
        <button id="disconnect-button" class="metamask-button" style="display:none;">Disconnect</button>
    </div>

    <style>
        #metamask-connector {
            border: 1px solid #ccc;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            background-color: #f9f9f9;
        }
        .metamask-button {
            background-color: #F6851B;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
        }
        .metamask-button:hover {
            background-color: #E2761B;
        }
        #metamask-status {
            margin-bottom: 10px;
        }
    </style>

    <script>
        // Function to check if MetaMask is installed
        function isMetaMaskInstalled() {
            return Boolean(window.ethereum && window.ethereum.isMetaMask);
        }

        // Function to update connection status
        function updateConnectionStatus(connected, address = null, networkName = null) {
            const statusElement = document.getElementById('connection-status');
            const addressElement = document.getElementById('wallet-address');
            const networkElement = document.getElementById('network-name');
            const connectButton = document.getElementById('connect-button');
            const disconnectButton = document.getElementById('disconnect-button');
            
            if (connected) {
                statusElement.textContent = 'MetaMask Status: Connected';
                statusElement.style.color = 'green';
                addressElement.textContent = `Wallet: ${address}`;
                networkElement.textContent = `Network: ${networkName}`;
                connectButton.style.display = 'none';
                disconnectButton.style.display = 'inline-block';
                
                // Send connection info to Streamlit
                const data = {
                    connected: true,
                    address: address,
                    networkName: networkName,
                    networkId: window.ethereum.networkVersion
                };
                window.parent.postMessage({
                    type: "metamask-status",
                    data: data
                }, "*");
            } else {
                statusElement.textContent = 'MetaMask Status: Not Connected';
                statusElement.style.color = 'red';
                addressElement.textContent = 'Wallet: Not Connected';
                networkElement.textContent = 'Network: Unknown';
                connectButton.style.display = 'inline-block';
                disconnectButton.style.display = 'none';
                
                // Send disconnection info to Streamlit
                window.parent.postMessage({
                    type: "metamask-status",
                    data: { connected: false }
                }, "*");
            }
        }

        // Function to get network name
        async function getNetworkName(chainId) {
            const networks = {
                '1': 'Ethereum Mainnet',
                '5': 'Goerli Testnet',
                '11155111': 'Sepolia Testnet',
                '137': 'Polygon Mainnet',
                '80001': 'Mumbai Testnet',
                '56': 'Binance Smart Chain',
                '97': 'BSC Testnet',
                '42161': 'Arbitrum One',
                '421613': 'Arbitrum Goerli',
                '10': 'Optimism',
                '420': 'Optimism Goerli',
                '1337': 'Local Development Chain',
                '31337': 'Hardhat Network'
            };
            
            return networks[chainId] || `Chain ID: ${chainId}`;
        }

        // Function to handle MetaMask connection
        async function connectMetaMask() {
            if (!isMetaMaskInstalled()) {
                alert("MetaMask is not installed. Please install MetaMask and try again.");
                return;
            }

            try {
                // Request account access
                const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
                const networkId = await window.ethereum.request({ method: 'eth_chainId' });
                const networkName = await getNetworkName(parseInt(networkId, 16).toString());
                
                if (accounts.length > 0) {
                    updateConnectionStatus(true, accounts[0], networkName);
                }
            } catch (error) {
                console.error("Error connecting to MetaMask:", error);
                alert("Error connecting to MetaMask: " + error.message);
                updateConnectionStatus(false);
            }
        }

        // Handle disconnection (note: MetaMask doesn't have a disconnect method,
        // so this just updates the UI without actually disconnecting)
        function disconnectMetaMask() {
            updateConnectionStatus(false);
        }

        // Add event listeners
        document.addEventListener('DOMContentLoaded', function() {
            // Check if MetaMask is installed
            if (!isMetaMaskInstalled()) {
                document.getElementById('connection-status').textContent = 'MetaMask Status: Not Installed';
                document.getElementById('connect-button').disabled = true;
                document.getElementById('connect-button').title = "Please install MetaMask first";
            }
            
            // Connect button event listener
            document.getElementById('connect-button').addEventListener('click', connectMetaMask);
            
            // Disconnect button event listener
            document.getElementById('disconnect-button').addEventListener('click', disconnectMetaMask);
            
            // Listen for account changes
            if (window.ethereum) {
                window.ethereum.on('accountsChanged', function (accounts) {
                    if (accounts.length === 0) {
                        // User disconnected all accounts
                        updateConnectionStatus(false);
                    } else {
                        // Account changed
                        connectMetaMask();
                    }
                });
                
                // Listen for chain changes
                window.ethereum.on('chainChanged', function (chainId) {
                    // Chain changed, reconnect
                    connectMetaMask();
                });
            }
        });
    </script>
    """

def metamask_connector():
    """
    Component that renders the MetaMask connector UI and returns connection info.
    
    Returns:
        dict: Connection information if connected, None otherwise
    """
    # Generate a unique key for the component
    component_key = "metamask_connector"
    
    # Create a placeholder for the component if it doesn't exist
    if component_key not in st.session_state:
        st.session_state[component_key] = {
            "connected": False,
            "address": None,
            "network_name": None,
            "network_id": None
        }
    
    # Render the HTML component
    components.html(
        get_metamask_component_html(),
        height=200,
        scrolling=False
    )
    
    # JavaScript callback to update session state
    st.markdown("""
    <script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'metamask-status') {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: event.data.data,
                dataType: 'json'
            }, '*');
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # For demonstration - since we can't get real callbacks in this version,
    # let's add some manual controls for testing
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Simulate Connection"):
            st.session_state[component_key] = {
                "connected": True,
                "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                "network_name": "Ethereum Mainnet",
                "network_id": "1"
            }
            st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
    
    with col2:
        if st.button("Simulate Disconnection"):
            st.session_state[component_key] = {
                "connected": False,
                "address": None,
                "network_name": None,
                "network_id": None
            }
            st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
    
    return st.session_state[component_key]