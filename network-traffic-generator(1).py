#!/usr/bin/env python3
"""
Advanced Network Protocol Testing Suite
Professional-grade network testing tool with routing protocol support and packet crafting

LEGAL NOTICE:
This tool is intended for authorized network testing, research, and educational purposes only.
Users must have explicit permission to test target networks and must comply with all applicable laws.
Unauthorized network testing may violate Computer Fraud and Abuse Act and similar laws.

Requirements:
pip install tkinter matplotlib scapy paho-mqtt pymodbus pycoapclient pysnmp netifaces pillow netmiko paramiko

Author: Network Research Tool
License: Educational/Research Use Only
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import time
import json
import queue
from datetime import datetime, timedelta
import socket
import ipaddress
import sys
import struct
import binascii
from typing import Dict, List, Optional, Tuple, Any
import webbrowser
import subprocess
import tempfile
import os
import requests
import math

# Import matplotlib for real-time charts
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.animation as animation
    from matplotlib.dates import DateFormatter
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Network libraries
try:
    from scapy.all import *
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.l2 import Ether, ARP
    from scapy.layers.inet6 import IPv6
    from scapy.contrib.ospf import OSPF_Hdr, OSPF_Hello
    from scapy.contrib.eigrp import EIGRP_Hdr, EIGRP_Hello
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# HTTP requests for geolocation
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available. Geolocation disabled.")

try:
    import netmiko
    NETMIKO_AVAILABLE = True
except ImportError:
    NETMIKO_AVAILABLE = False

try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

try:
    from pymodbus.client.sync import ModbusTcpClient
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False

try:
    from pysnmp.hlapi import *
    SNMP_AVAILABLE = True
except ImportError:
    SNMP_AVAILABLE = False

try:
    import netifaces
    NETIFACES_AVAILABLE = True
except ImportError:
    NETIFACES_AVAILABLE = False


class ModernTheme:
    """Modern dark theme configuration for professional appearance"""
    
    # Color palette - Professional network tool styling
    DARK_BG = "#1e1e1e"
    DARKER_BG = "#161616"
    ACCENT_BG = "#2d2d2d"
    HIGHLIGHT_BG = "#404040"
    CARD_BG = "#333333"
    
    PRIMARY_COLOR = "#0078d4"
    SUCCESS_COLOR = "#16c60c"
    WARNING_COLOR = "#ff8c00"
    ERROR_COLOR = "#d13438"
    CISCO_BLUE = "#1ba1e2"
    JUNIPER_GREEN = "#84bd00"
    
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#b3b3b3"
    TEXT_DISABLED = "#666666"
    
    BORDER_COLOR = "#404040"
    
    @classmethod
    def configure_style(cls):
        """Configure ttk styles for modern professional appearance"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure Notebook (tabs) - Professional look
        style.configure('Modern.TNotebook', 
                       background=cls.DARK_BG,
                       borderwidth=0,
                       tabmargins=[2, 5, 2, 0])
        style.configure('Modern.TNotebook.Tab',
                       background=cls.ACCENT_BG,
                       foreground=cls.TEXT_PRIMARY,
                       padding=[15, 10],
                       borderwidth=1)
        style.map('Modern.TNotebook.Tab',
                 background=[('selected', cls.PRIMARY_COLOR),
                           ('active', cls.HIGHLIGHT_BG)])
        
        # Configure Frame
        style.configure('Modern.TFrame',
                       background=cls.DARK_BG,
                       relief='flat')
        style.configure('Card.TFrame',
                       background=cls.CARD_BG,
                       relief='solid',
                       borderwidth=1)
        
        # Configure Labels
        style.configure('Modern.TLabel',
                       background=cls.DARK_BG,
                       foreground=cls.TEXT_PRIMARY,
                       font=('Segoe UI', 9))
        style.configure('Title.TLabel',
                       background=cls.DARK_BG,
                       foreground=cls.TEXT_PRIMARY,
                       font=('Segoe UI', 14, 'bold'))
        style.configure('Header.TLabel',
                       background=cls.DARK_BG,
                       foreground=cls.PRIMARY_COLOR,
                       font=('Segoe UI', 11, 'bold'))
        style.configure('SubHeader.TLabel',
                       background=cls.DARK_BG,
                       foreground=cls.TEXT_SECONDARY,
                       font=('Segoe UI', 10, 'bold'))
        
        # Configure Entry
        style.configure('Modern.TEntry',
                       fieldbackground=cls.ACCENT_BG,
                       background=cls.ACCENT_BG,
                       foreground=cls.TEXT_PRIMARY,
                       borderwidth=1,
                       insertcolor=cls.TEXT_PRIMARY)
        
        # Configure Combobox
        style.configure('Modern.TCombobox',
                       fieldbackground=cls.ACCENT_BG,
                       background=cls.ACCENT_BG,
                       foreground=cls.TEXT_PRIMARY,
                       borderwidth=1)
        
        # Configure Button
        style.configure('Modern.TButton',
                       background=cls.PRIMARY_COLOR,
                       foreground=cls.TEXT_PRIMARY,
                       borderwidth=0,
                       focuscolor='none',
                       padding=[12, 8])
        style.map('Modern.TButton',
                 background=[('active', '#106ebe'),
                           ('pressed', '#005a9e')])
        
        # Configure Cisco/Juniper specific buttons
        style.configure('Cisco.TButton',
                       background=cls.CISCO_BLUE,
                       foreground=cls.TEXT_PRIMARY,
                       borderwidth=0,
                       focuscolor='none',
                       padding=[12, 8])
        style.configure('Juniper.TButton',
                       background=cls.JUNIPER_GREEN,
                       foreground=cls.TEXT_PRIMARY,
                       borderwidth=0,
                       focuscolor='none',
                       padding=[12, 8])
        
        # Configure Success/Warning/Error buttons
        style.configure('Success.TButton',
                       background=cls.SUCCESS_COLOR,
                       foreground=cls.TEXT_PRIMARY)
        style.configure('Warning.TButton',
                       background=cls.WARNING_COLOR,
                       foreground=cls.TEXT_PRIMARY)
        style.configure('Error.TButton',
                       background=cls.ERROR_COLOR,
                       foreground=cls.TEXT_PRIMARY)
        
        # Configure Treeview
        style.configure('Modern.Treeview',
                       background=cls.ACCENT_BG,
                       foreground=cls.TEXT_PRIMARY,
                       fieldbackground=cls.ACCENT_BG,
                       borderwidth=0)
        style.configure('Modern.Treeview.Heading',
                       background=cls.DARKER_BG,
                       foreground=cls.TEXT_PRIMARY,
                       borderwidth=1)
        
        # Configure Progressbar
        style.configure('Modern.Horizontal.TProgressbar',
                       background=cls.PRIMARY_COLOR,
                       troughcolor=cls.ACCENT_BG,
                       borderwidth=0)


class StatusIndicator(tk.Frame):
    """Professional status indicator widget"""
    
    def __init__(self, parent, text="Status", **kwargs):
        super().__init__(parent, bg=ModernTheme.DARK_BG, **kwargs)
        
        self.status_color = ModernTheme.TEXT_DISABLED
        
        # Create indicator circle
        self.canvas = tk.Canvas(self, width=12, height=12, 
                               bg=ModernTheme.DARK_BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=(0, 8))
        
        self.circle = self.canvas.create_oval(2, 2, 10, 10, 
                                            fill=self.status_color, outline="")
        
        # Status label
        self.label = tk.Label(self, text=text, 
                             bg=ModernTheme.DARK_BG, 
                             fg=ModernTheme.TEXT_PRIMARY,
                             font=('Segoe UI', 9))
        self.label.pack(side=tk.LEFT)
    
    def set_status(self, status: str, color: str = None):
        """Update status indicator"""
        if color:
            self.status_color = color
        else:
            color_map = {
                'active': ModernTheme.SUCCESS_COLOR,
                'inactive': ModernTheme.TEXT_DISABLED,
                'warning': ModernTheme.WARNING_COLOR,
                'error': ModernTheme.ERROR_COLOR,
                'connected': ModernTheme.SUCCESS_COLOR,
                'disconnected': ModernTheme.TEXT_DISABLED
            }
            self.status_color = color_map.get(status.lower(), ModernTheme.TEXT_DISABLED)
        
        self.canvas.itemconfig(self.circle, fill=self.status_color)
        self.label.config(text=status.title())


class MetricCard(tk.Frame):
    """Professional metric display card"""
    
    def __init__(self, parent, title="Metric", value="0", unit="", icon="📊", **kwargs):
        super().__init__(parent, bg=ModernTheme.CARD_BG, 
                        relief='solid', bd=1, **kwargs)
        
        # Header with icon
        header_frame = tk.Frame(self, bg=ModernTheme.CARD_BG)
        header_frame.pack(fill='x', padx=15, pady=(15, 5))
        
        icon_label = tk.Label(header_frame, text=icon, 
                             bg=ModernTheme.CARD_BG,
                             fg=ModernTheme.PRIMARY_COLOR,
                             font=('Segoe UI', 14))
        icon_label.pack(side='left')
        
        title_label = tk.Label(header_frame, text=title, 
                              bg=ModernTheme.CARD_BG,
                              fg=ModernTheme.TEXT_SECONDARY,
                              font=('Segoe UI', 10, 'bold'))
        title_label.pack(side='left', padx=(10, 0))
        
        # Value display
        value_frame = tk.Frame(self, bg=ModernTheme.CARD_BG)
        value_frame.pack(padx=15, pady=(0, 15))
        
        self.value_label = tk.Label(value_frame, text=value,
                                   bg=ModernTheme.CARD_BG,
                                   fg=ModernTheme.TEXT_PRIMARY,
                                   font=('Segoe UI', 18, 'bold'))
        self.value_label.pack(side='left')
        
        if unit:
            unit_label = tk.Label(value_frame, text=unit,
                                 bg=ModernTheme.CARD_BG,
                                 fg=ModernTheme.TEXT_SECONDARY,
                                 font=('Segoe UI', 12))
            unit_label.pack(side='left', padx=(5, 0))
    
    def update_value(self, value: str):
        """Update the metric value"""
        self.value_label.config(text=value)


class PacketCraftingWidget(tk.Frame):
    """Advanced packet crafting interface"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=ModernTheme.DARK_BG, **kwargs)
        
        self.packet_layers = []
        self.current_packet = None
        self.layer_configs = {}
        self.create_interface()
    
    def create_interface(self):
        """Create packet crafting interface"""
        # Header
        header_frame = tk.Frame(self, bg=ModernTheme.DARK_BG)
        header_frame.pack(fill='x', pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="🔧 Advanced Packet Crafter", 
                               style='Title.TLabel')
        title_label.pack(side='left')
        
        # Layer management buttons
        button_frame = tk.Frame(header_frame, bg=ModernTheme.DARK_BG)
        button_frame.pack(side='right')
        
        ttk.Button(button_frame, text="Clear All", 
                  style='Warning.TButton',
                  command=self.clear_layers).pack(side='left', padx=(0, 5))
        ttk.Button(button_frame, text="Build Packet", 
                  style='Modern.TButton',
                  command=self.build_packet).pack(side='left', padx=(0, 5))
        ttk.Button(button_frame, text="Send Packet", 
                  style='Success.TButton',
                  command=self.send_packet).pack(side='left')
        
        # Main content area
        main_frame = tk.Frame(self, bg=ModernTheme.DARK_BG)
        main_frame.pack(fill='both', expand=True)
        
        # Left panel - Layer selection
        left_panel = tk.Frame(main_frame, bg=ModernTheme.ACCENT_BG, width=250)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        ttk.Label(left_panel, text="Protocol Layers", 
                 style='Header.TLabel').pack(pady=15)
        
        # Layer categories
        self.create_layer_categories(left_panel)
        
        # Right panel - Packet construction
        right_panel = tk.Frame(main_frame, bg=ModernTheme.DARK_BG)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Packet layers display
        layers_label = ttk.Label(right_panel, text="Packet Layers", 
                                style='Header.TLabel')
        layers_label.pack(anchor='w', pady=(0, 10))
        
        # Layer list
        self.layer_listbox = tk.Listbox(right_panel,
                                       bg=ModernTheme.ACCENT_BG,
                                       fg=ModernTheme.TEXT_PRIMARY,
                                       selectbackground=ModernTheme.PRIMARY_COLOR,
                                       borderwidth=0,
                                       highlightthickness=0,
                                       font=('Segoe UI', 10),
                                       height=6)
        self.layer_listbox.pack(fill='x', pady=(0, 10))
        self.layer_listbox.bind('<<ListboxSelect>>', self.on_layer_select)
        
        # Remove layer button
        remove_btn = ttk.Button(right_panel, text="Remove Selected Layer",
                               style='Error.TButton',
                               command=self.remove_selected_layer)
        remove_btn.pack(anchor='w', pady=(0, 20))
        
        # Packet preview
        preview_label = ttk.Label(right_panel, text="Packet Preview", 
                                 style='Header.TLabel')
        preview_label.pack(anchor='w', pady=(0, 10))
        
        self.packet_preview = scrolledtext.ScrolledText(right_panel,
                                                       bg=ModernTheme.DARKER_BG,
                                                       fg=ModernTheme.TEXT_PRIMARY,
                                                       font=('Consolas', 10),
                                                       height=8)
        self.packet_preview.pack(fill='x', pady=(0, 20))
        
        # Layer configuration
        config_label = ttk.Label(right_panel, text="Layer Configuration", 
                                style='Header.TLabel')
        config_label.pack(anchor='w', pady=(0, 10))
        
        self.config_frame = tk.Frame(right_panel, bg=ModernTheme.DARK_BG)
        self.config_frame.pack(fill='both', expand=True)
        
        # Initial message
        initial_label = tk.Label(self.config_frame, 
                                text="Select a protocol layer to begin packet crafting",
                                bg=ModernTheme.DARK_BG,
                                fg=ModernTheme.TEXT_SECONDARY,
                                font=('Segoe UI', 12))
        initial_label.pack(expand=True)
    
    def create_layer_categories(self, parent):
        """Create layer selection categories"""
        categories = {
            "Layer 2": ["Ethernet", "ARP", "802.1Q VLAN"],
            "Layer 3": ["IPv4", "IPv6", "ICMP", "ICMPv6"],
            "Layer 4": ["TCP", "UDP", "SCTP"],
            "Routing": ["BGP", "OSPF", "EIGRP", "RIP"],
            "Application": ["HTTP", "DNS", "DHCP"],
            "Custom": ["Raw Payload", "Hex Data"]
        }
        
        for category, protocols in categories.items():
            # Category header
            cat_frame = tk.Frame(parent, bg=ModernTheme.ACCENT_BG)
            cat_frame.pack(fill='x', padx=10, pady=(10, 5))
            
            cat_label = tk.Label(cat_frame, text=category,
                                bg=ModernTheme.ACCENT_BG,
                                fg=ModernTheme.PRIMARY_COLOR,
                                font=('Segoe UI', 10, 'bold'))
            cat_label.pack(anchor='w')
            
            # Protocol buttons
            for protocol in protocols:
                protocol_btn = tk.Button(parent, text=protocol,
                                       bg=ModernTheme.HIGHLIGHT_BG,
                                       fg=ModernTheme.TEXT_PRIMARY,
                                       font=('Segoe UI', 9),
                                       relief='flat',
                                       cursor='hand2',
                                       command=lambda p=protocol: self.select_protocol(p))
                protocol_btn.pack(fill='x', padx=15, pady=1)
                
                # Hover effects
                def on_enter(e, btn=protocol_btn):
                    btn.config(bg=ModernTheme.PRIMARY_COLOR)
                def on_leave(e, btn=protocol_btn):
                    btn.config(bg=ModernTheme.HIGHLIGHT_BG)
                
                protocol_btn.bind("<Enter>", on_enter)
                protocol_btn.bind("<Leave>", on_leave)
    
    def select_protocol(self, protocol):
        """Handle protocol selection"""
        # Clear existing configuration
        for widget in self.config_frame.winfo_children():
            widget.destroy()
        
        # Create protocol-specific configuration
        self.create_protocol_config(protocol)
    
    def create_protocol_config(self, protocol):
        """Create configuration interface for selected protocol"""
        # Protocol header
        header_frame = tk.Frame(self.config_frame, bg=ModernTheme.DARK_BG)
        header_frame.pack(fill='x', pady=(0, 20))
        
        protocol_label = ttk.Label(header_frame, text=f"{protocol} Configuration", 
                                  style='SubHeader.TLabel')
        protocol_label.pack(side='left')
        
        add_btn = ttk.Button(header_frame, text="Add to Packet",
                            style='Success.TButton',
                            command=lambda: self.add_protocol_layer(protocol))
        add_btn.pack(side='right')
        
        # Configuration fields based on protocol
        if protocol == "IPv4":
            self.create_ipv4_config()
        elif protocol == "TCP":
            self.create_tcp_config()
        elif protocol == "UDP":
            self.create_udp_config()
        elif protocol == "Ethernet":
            self.create_ethernet_config()
        elif protocol == "ICMP":
            self.create_icmp_config()
        elif protocol == "ARP":
            self.create_arp_config()
        elif protocol == "DNS":
            self.create_dns_config()
        elif protocol == "DHCP":
            self.create_dhcp_config()
        elif protocol == "Raw Payload":
            self.create_raw_payload_config()
        elif protocol == "Hex Data":
            self.create_hex_data_config()
        else:
            # Generic configuration
            self.create_generic_config(protocol)
    
    def create_ipv4_config(self):
        """Create IPv4 configuration fields"""
        config_frame = tk.Frame(self.config_frame, bg=ModernTheme.DARK_BG)
        config_frame.pack(fill='x', pady=10)
        
        # Source IP
        self.create_field(config_frame, "Source IP:", "192.168.1.100", "ipv4_src")
        self.create_field(config_frame, "Destination IP:", "192.168.1.1", "ipv4_dst")
        self.create_field(config_frame, "TTL:", "64", "ipv4_ttl")
        self.create_field(config_frame, "Protocol:", "1", "ipv4_proto")
        self.create_field(config_frame, "TOS:", "0", "ipv4_tos")
        self.create_field(config_frame, "ID:", "1", "ipv4_id")
    
    def create_tcp_config(self):
        """Create TCP configuration fields"""
        config_frame = tk.Frame(self.config_frame, bg=ModernTheme.DARK_BG)
        config_frame.pack(fill='x', pady=10)
        
        self.create_field(config_frame, "Source Port:", "12345", "tcp_sport")
        self.create_field(config_frame, "Destination Port:", "80", "tcp_dport")
        self.create_field(config_frame, "Sequence Number:", "1000", "tcp_seq")
        self.create_field(config_frame, "Acknowledgment:", "0", "tcp_ack")
        self.create_field(config_frame, "Flags:", "S", "tcp_flags")
        self.create_field(config_frame, "Window Size:", "8192", "tcp_window")
    
    def create_udp_config(self):
        """Create UDP configuration fields"""
        config_frame = tk.Frame(self.config_frame, bg=ModernTheme.DARK_BG)
        config_frame.pack(fill='x', pady=10)
        
        self.create_field(config_frame, "Source Port:", "12345", "udp_sport")
        self.create_field(config_frame, "Destination Port:", "53", "udp_dport")
    
    def create_ethernet_config(self):
        """Create Ethernet configuration fields"""
        config_frame = tk.Frame(self.config_frame, bg=ModernTheme.DARK_BG)
        config_frame.pack(fill='x', pady=10)
        
        self.create_field(config_frame, "Source MAC:", "00:11:22:33:44:55", "eth_src")
        self.create_field(config_frame, "Destination MAC:", "ff:ff:ff:ff:ff:ff", "eth_dst")
        self.create_field(config_frame, "EtherType:", "0x0800", "eth_type")
    
    def create_icmp_config(self):
        """Create ICMP configuration fields"""
        config_frame = tk.Frame(self.config_frame, bg=ModernTheme.DARK_BG)
        config_frame.pack(fill='x', pady=10)
        
        self.create_field(config_frame, "Type:", "8", "icmp_type")
        self.create_field(config_frame, "Code:", "0", "icmp_code")
        self.create_field(config_frame, "ID:", "1", "icmp_id")
        self.create_field(config_frame, "Sequence:", "1", "icmp_seq")
    
    def create_arp_config(self):
        """Create ARP configuration fields"""
        config_frame = tk.Frame(self.config_frame, bg=ModernTheme.DARK_BG)
        config_frame.pack(fill='x', pady=10)
        
        self.create_field(config_frame, "Operation:", "1", "arp_op")  # 1=request, 2=reply
        self.create_field(config_frame, "Sender MAC:", "00:11:22:33:44:55", "arp_hwsrc")
        self.create_field(config_frame, "Sender IP:", "192.168.1.100", "arp_psrc")
        self.create_field(config_frame, "Target MAC:", "00:00:00:00:00:00", "arp_hwdst")
        self.create_field(config_frame, "Target IP:", "192.168.1.1", "arp_pdst")
    
    def create_dns_config(self):
        """Create DNS configuration fields"""
        config_frame = tk.Frame(self.config_frame, bg=ModernTheme.DARK_BG)
        config_frame.pack(fill='x', pady=10)
        
        self.create_field(config_frame, "Query Name:", "example.com", "dns_qname")
        self.create_field(config_frame, "Query Type:", "A", "dns_qtype")
        self.create_field(config_frame, "Query Class:", "IN", "dns_qclass")
        self.create_field(config_frame, "Recursion Desired:", "1", "dns_rd")
    
    def create_dhcp_config(self):
        """Create DHCP configuration fields"""
        config_frame = tk.Frame(self.config_frame, bg=ModernTheme.DARK_BG)
        config_frame.pack(fill='x', pady=10)
        
        self.create_field(config_frame, "Message Type:", "discover", "dhcp_msg_type")
        self.create_field(config_frame, "Client MAC:", "00:11:22:33:44:55", "dhcp_chaddr")
        self.create_field(config_frame, "Requested IP:", "0.0.0.0", "dhcp_req_ip")
    
    def create_raw_payload_config(self):
        """Create raw payload configuration"""
        config_frame = tk.Frame(self.config_frame, bg=ModernTheme.DARK_BG)
        config_frame.pack(fill='both', expand=True, pady=10)
        
        ttk.Label(config_frame, text="Raw Payload Data:", 
                 style='Modern.TLabel').pack(anchor='w', pady=(0, 5))
        
        self.raw_payload_text = scrolledtext.ScrolledText(config_frame,
                                                         bg=ModernTheme.ACCENT_BG,
                                                         fg=ModernTheme.TEXT_PRIMARY,
                                                         font=('Consolas', 10),
                                                         height=6)
        self.raw_payload_text.pack(fill='both', expand=True)
        self.raw_payload_text.insert(tk.END, "Hello, World! This is a test payload.")
    
    def create_hex_data_config(self):
        """Create hex data configuration"""
        config_frame = tk.Frame(self.config_frame, bg=ModernTheme.DARK_BG)
        config_frame.pack(fill='both', expand=True, pady=10)
        
        ttk.Label(config_frame, text="Hex Data (space separated):", 
                 style='Modern.TLabel').pack(anchor='w', pady=(0, 5))
        
        self.hex_data_text = scrolledtext.ScrolledText(config_frame,
                                                      bg=ModernTheme.ACCENT_BG,
                                                      fg=ModernTheme.TEXT_PRIMARY,
                                                      font=('Consolas', 10),
                                                      height=6)
        self.hex_data_text.pack(fill='both', expand=True)
        self.hex_data_text.insert(tk.END, "48 65 6c 6c 6f 20 57 6f 72 6c 64")
    
    def create_generic_config(self, protocol):
        """Create generic configuration for unsupported protocols"""
        config_frame = tk.Frame(self.config_frame, bg=ModernTheme.DARK_BG)
        config_frame.pack(fill='x', pady=10)
        
        info_label = tk.Label(config_frame, 
                             text=f"{protocol} protocol configuration not yet implemented.\nUse Raw Payload for custom data.",
                             bg=ModernTheme.DARK_BG,
                             fg=ModernTheme.TEXT_SECONDARY,
                             font=('Segoe UI', 10))
        info_label.pack(pady=20)
    
    def create_field(self, parent, label, default_value, field_name):
        """Create a configuration field"""
        field_frame = tk.Frame(parent, bg=ModernTheme.DARK_BG)
        field_frame.pack(fill='x', pady=5)
        
        label_widget = ttk.Label(field_frame, text=label, style='Modern.TLabel')
        label_widget.pack(side='left', anchor='w')
        
        entry_var = tk.StringVar(value=default_value)
        entry_widget = ttk.Entry(field_frame, textvariable=entry_var, 
                               style='Modern.TEntry', width=20)
        entry_widget.pack(side='right')
        
        # Store reference for later use
        setattr(self, f"{field_name}_var", entry_var)
    
    def add_protocol_layer(self, protocol):
        """Add protocol layer to packet"""
        # Get configuration values
        config = {}
        
        # Collect configuration based on protocol
        if protocol == "IPv4":
            config = {
                'src': getattr(self, 'ipv4_src_var', tk.StringVar()).get(),
                'dst': getattr(self, 'ipv4_dst_var', tk.StringVar()).get(),
                'ttl': int(getattr(self, 'ipv4_ttl_var', tk.StringVar(value='64')).get()),
                'proto': int(getattr(self, 'ipv4_proto_var', tk.StringVar(value='1')).get()),
                'tos': int(getattr(self, 'ipv4_tos_var', tk.StringVar(value='0')).get()),
                'id': int(getattr(self, 'ipv4_id_var', tk.StringVar(value='1')).get())
            }
        elif protocol == "TCP":
            config = {
                'sport': int(getattr(self, 'tcp_sport_var', tk.StringVar(value='12345')).get()),
                'dport': int(getattr(self, 'tcp_dport_var', tk.StringVar(value='80')).get()),
                'seq': int(getattr(self, 'tcp_seq_var', tk.StringVar(value='1000')).get()),
                'ack': int(getattr(self, 'tcp_ack_var', tk.StringVar(value='0')).get()),
                'flags': getattr(self, 'tcp_flags_var', tk.StringVar(value='S')).get(),
                'window': int(getattr(self, 'tcp_window_var', tk.StringVar(value='8192')).get())
            }
        elif protocol == "UDP":
            config = {
                'sport': int(getattr(self, 'udp_sport_var', tk.StringVar(value='12345')).get()),
                'dport': int(getattr(self, 'udp_dport_var', tk.StringVar(value='53')).get())
            }
        elif protocol == "Ethernet":
            config = {
                'src': getattr(self, 'eth_src_var', tk.StringVar()).get(),
                'dst': getattr(self, 'eth_dst_var', tk.StringVar()).get(),
                'type': getattr(self, 'eth_type_var', tk.StringVar()).get()
            }
        elif protocol == "ICMP":
            config = {
                'type': int(getattr(self, 'icmp_type_var', tk.StringVar(value='8')).get()),
                'code': int(getattr(self, 'icmp_code_var', tk.StringVar(value='0')).get()),
                'id': int(getattr(self, 'icmp_id_var', tk.StringVar(value='1')).get()),
                'seq': int(getattr(self, 'icmp_seq_var', tk.StringVar(value='1')).get())
            }
        elif protocol == "Raw Payload":
            config = {
                'load': getattr(self, 'raw_payload_text', None).get(1.0, tk.END).strip() if hasattr(self, 'raw_payload_text') else "Hello World"
            }
        elif protocol == "Hex Data":
            hex_string = getattr(self, 'hex_data_text', None).get(1.0, tk.END).strip() if hasattr(self, 'hex_data_text') else "48656c6c6f"
            try:
                # Convert hex string to bytes
                hex_bytes = bytes.fromhex(hex_string.replace(' ', ''))
                config = {'load': hex_bytes}
            except ValueError:
                messagebox.showerror("Error", "Invalid hex data format")
                return
        
        # Add layer with configuration
        layer_data = {
            'protocol': protocol,
            'config': config
        }
        
        self.packet_layers.append(layer_data)
        self.layer_configs[protocol] = config
        
        # Update displays
        self.update_layer_list()
        self.update_packet_preview()
        
        # Clear configuration
        for widget in self.config_frame.winfo_children():
            widget.destroy()
        
        initial_label = tk.Label(self.config_frame, 
                                text="Layer added! Select another protocol or build the packet.",
                                bg=ModernTheme.DARK_BG,
                                fg=ModernTheme.SUCCESS_COLOR,
                                font=('Segoe UI', 12))
        initial_label.pack(expand=True)
    
    def update_layer_list(self):
        """Update the layer list display"""
        self.layer_listbox.delete(0, tk.END)
        for i, layer_data in enumerate(self.packet_layers):
            self.layer_listbox.insert(tk.END, f"{i+1}. {layer_data['protocol']}")
    
    def on_layer_select(self, event):
        """Handle layer selection in listbox"""
        selection = self.layer_listbox.curselection()
        if selection:
            layer_index = selection[0]
            layer_data = self.packet_layers[layer_index]
            
            # Show layer configuration in preview
            config_text = f"Layer {layer_index + 1}: {layer_data['protocol']}\n"
            config_text += "Configuration:\n"
            for key, value in layer_data['config'].items():
                config_text += f"  {key}: {value}\n"
            
            self.packet_preview.delete(1.0, tk.END)
            self.packet_preview.insert(tk.END, config_text)
    
    def remove_selected_layer(self):
        """Remove selected layer"""
        selection = self.layer_listbox.curselection()
        if selection:
            layer_index = selection[0]
            del self.packet_layers[layer_index]
            self.update_layer_list()
            self.update_packet_preview()
    
    def clear_layers(self):
        """Clear all packet layers"""
        self.packet_layers = []
        self.layer_configs = {}
        self.current_packet = None
        self.update_layer_list()
        self.update_packet_preview()
        
        # Clear configuration
        for widget in self.config_frame.winfo_children():
            widget.destroy()
        
        initial_label = tk.Label(self.config_frame, 
                                text="All layers cleared. Select a protocol to begin.",
                                bg=ModernTheme.DARK_BG,
                                fg=ModernTheme.TEXT_SECONDARY,
                                font=('Segoe UI', 12))
        initial_label.pack(expand=True)
    
    def build_packet(self):
        """Build the actual packet from layers"""
        if not self.packet_layers:
            messagebox.showwarning("Warning", "No layers configured for packet")
            return
        
        if not SCAPY_AVAILABLE:
            messagebox.showerror("Error", "Scapy is required for packet building")
            return
        
        try:
            packet = None
            
            # Build packet layer by layer
            for layer_data in self.packet_layers:
                protocol = layer_data['protocol']
                config = layer_data['config']
                
                if protocol == "Ethernet":
                    layer = Ether(**config)
                elif protocol == "IPv4":
                    layer = IP(**config)
                elif protocol == "TCP":
                    layer = TCP(**config)
                elif protocol == "UDP":
                    layer = UDP(**config)
                elif protocol == "ICMP":
                    layer = ICMP(**config)
                elif protocol == "ARP":
                    layer = ARP(**config)
                elif protocol == "Raw Payload":
                    layer = Raw(**config)
                elif protocol == "Hex Data":
                    layer = Raw(**config)
                else:
                    # Skip unsupported protocols
                    continue
                
                # Stack layers
                if packet is None:
                    packet = layer
                else:
                    packet = packet / layer
            
            self.current_packet = packet
            
            # Update preview with built packet
            self.packet_preview.delete(1.0, tk.END)
            self.packet_preview.insert(tk.END, "Built Packet Summary:\n")
            self.packet_preview.insert(tk.END, "=" * 50 + "\n\n")
            
            if packet:
                self.packet_preview.insert(tk.END, packet.summary() + "\n\n")
                self.packet_preview.insert(tk.END, "Packet Details:\n")
                self.packet_preview.insert(tk.END, str(packet))
            
            messagebox.showinfo("Success", f"Packet built successfully with {len(self.packet_layers)} layers!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to build packet: {str(e)}")
    
    def update_packet_preview(self):
        """Update the packet preview display"""
        self.packet_preview.delete(1.0, tk.END)
        
        if not self.packet_layers:
            self.packet_preview.insert(tk.END, "No layers added yet.\n\nSelect protocols from the left panel to build your packet.")
            return
        
        preview_text = "Packet Structure:\n"
        preview_text += "=" * 50 + "\n\n"
        
        for i, layer_data in enumerate(self.packet_layers, 1):
            preview_text += f"Layer {i}: {layer_data['protocol']}\n"
            for key, value in layer_data['config'].items():
                preview_text += f"  {key}: {value}\n"
            preview_text += "\n"
        
        preview_text += "=" * 50 + "\n"
        preview_text += "Click 'Build Packet' to construct the actual packet.\n"
        preview_text += "Click 'Send Packet' to transmit the built packet."
        
        self.packet_preview.insert(tk.END, preview_text)
    
    def send_packet(self):
        """Send the crafted packet"""
        if not self.current_packet:
            # Try to build packet first
            self.build_packet()
            
        if not self.current_packet:
            messagebox.showwarning("Warning", "No packet built. Please build packet first.")
            return
        
        if not SCAPY_AVAILABLE:
            messagebox.showerror("Error", "Scapy is required for packet sending")
            return
        
        # Show confirmation dialog
        if not messagebox.askyesno("Confirm Send", 
                                  f"Send packet with {len(self.packet_layers)} layers?\n\n"
                                  f"Packet: {self.current_packet.summary()}\n\n"
                                  f"⚠️ Ensure you have authorization to send this packet!"):
            return
        
        try:
            # Determine if we need to use send() or sendp()
            if self.current_packet.haslayer(Ether):
                # Has Ethernet layer, use sendp (Layer 2)
                sendp(self.current_packet, verbose=False)
                send_type = "Layer 2 (sendp)"
            else:
                # No Ethernet layer, use send (Layer 3)
                send(self.current_packet, verbose=False)
                send_type = "Layer 3 (send)"
            
            messagebox.showinfo("Success", 
                               f"Packet sent successfully!\n\n"
                               f"Method: {send_type}\n"
                               f"Layers: {len(self.packet_layers)}\n"
                               f"Size: {len(self.current_packet)} bytes")
            
            # Log the packet send
            if hasattr(self.master.master.master, 'add_log'):
                self.master.master.master.add_log(
                    f"Custom packet sent: {self.current_packet.summary()}", "SUCCESS")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send packet: {str(e)}")
            
class NetworkPathDiscovery(tk.Frame):
    """Advanced network path discovery with geolocation and hop response management"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=ModernTheme.DARK_BG, **kwargs)
        
        self.hops = []
        self.hop_responses = {}
        self.target_host = ""
        self.is_probing = False
        self.map_canvas = None
        self.hop_locations = {}
        self.selected_hop = None
        
        self.create_interface()
    
    def create_interface(self):
        """Create network path discovery interface"""
        # Header
        header_frame = tk.Frame(self, bg=ModernTheme.DARK_BG)
        header_frame.pack(fill='x', pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="🗺️ Network Path Discovery & Hop Management", 
                               style='Title.TLabel')
        title_label.pack(side='left')
        
        # Control buttons
        button_frame = tk.Frame(header_frame, bg=ModernTheme.DARK_BG)
        button_frame.pack(side='right')
        
        ttk.Button(button_frame, text="Clear Route", 
                  style='Warning.TButton',
                  command=self.clear_route).pack(side='left', padx=(0, 5))
        ttk.Button(button_frame, text="Export Route", 
                  style='Modern.TButton',
                  command=self.export_route).pack(side='left', padx=(0, 5))
        ttk.Button(button_frame, text="Import Route", 
                  style='Modern.TButton',
                  command=self.import_route).pack(side='left')
        
        # Main content area
        main_frame = tk.Frame(self, bg=ModernTheme.DARK_BG)
        main_frame.pack(fill='both', expand=True)
        
        # Left panel - Controls and hop list
        left_panel = tk.Frame(main_frame, bg=ModernTheme.CARD_BG, width=350)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        self.create_probe_controls(left_panel)
        self.create_hop_list(left_panel)
        
        # Right panel - Network map
        right_panel = tk.Frame(main_frame, bg=ModernTheme.DARK_BG)
        right_panel.pack(side='right', fill='both', expand=True)
        
        self.create_network_map(right_panel)
    
    def create_probe_controls(self, parent):
        """Create probe control interface"""
        # Controls header
        controls_header = tk.Frame(parent, bg=ModernTheme.CARD_BG)
        controls_header.pack(fill='x', padx=15, pady=15)
        
        ttk.Label(controls_header, text="🎯 Route Discovery", 
                 style='Header.TLabel').pack(side='left')
        
        self.probe_status = StatusIndicator(controls_header, "Idle")
        self.probe_status.pack(side='right')
        
        # Target configuration
        target_frame = tk.Frame(parent, bg=ModernTheme.CARD_BG)
        target_frame.pack(fill='x', padx=15, pady=(0, 10))
        
        ttk.Label(target_frame, text="Target Host:", style='Modern.TLabel').pack(anchor='w', pady=(0, 5))
        self.target_var = tk.StringVar(value="8.8.8.8")
        ttk.Entry(target_frame, textvariable=self.target_var, 
                 style='Modern.TEntry').pack(fill='x', pady=(0, 10))
        
        # Probe options
        options_frame = tk.Frame(target_frame, bg=ModernTheme.CARD_BG)
        options_frame.pack(fill='x', pady=(0, 10))
        
        # Max hops
        max_hops_frame = tk.Frame(options_frame, bg=ModernTheme.CARD_BG)
        max_hops_frame.pack(fill='x', pady=2)
        
        ttk.Label(max_hops_frame, text="Max Hops:", style='Modern.TLabel').pack(side='left')
        self.max_hops_var = tk.StringVar(value="30")
        ttk.Entry(max_hops_frame, textvariable=self.max_hops_var, 
                 style='Modern.TEntry', width=8).pack(side='right')
        
        # Timeout
        timeout_frame = tk.Frame(options_frame, bg=ModernTheme.CARD_BG)
        timeout_frame.pack(fill='x', pady=2)
        
        ttk.Label(timeout_frame, text="Timeout (s):", style='Modern.TLabel').pack(side='left')
        self.timeout_var = tk.StringVar(value="3")
        ttk.Entry(timeout_frame, textvariable=self.timeout_var, 
                 style='Modern.TEntry', width=8).pack(side='right')
        
        # Probe type
        probe_type_frame = tk.Frame(options_frame, bg=ModernTheme.CARD_BG)
        probe_type_frame.pack(fill='x', pady=2)
        
        ttk.Label(probe_type_frame, text="Probe Type:", style='Modern.TLabel').pack(side='left')
        self.probe_type_var = tk.StringVar(value="ICMP")
        probe_combo = ttk.Combobox(probe_type_frame, textvariable=self.probe_type_var,
                                  values=["ICMP", "UDP", "TCP"],
                                  style='Modern.TCombobox', width=10, state='readonly')
        probe_combo.pack(side='right')
        
        # Start probe button
        self.start_probe_btn = ttk.Button(target_frame, text="🚀 Start Route Discovery",
                                         style='Success.TButton',
                                         command=self.start_route_discovery)
        self.start_probe_btn.pack(fill='x', pady=(10, 0))
        
        self.stop_probe_btn = ttk.Button(target_frame, text="⏹ Stop Discovery",
                                        style='Error.TButton',
                                        command=self.stop_route_discovery,
                                        state='disabled')
        self.stop_probe_btn.pack(fill='x', pady=(5, 0))
    
    def create_hop_list(self, parent):
        """Create hop list interface"""
        # Hop list header
        hop_header = tk.Frame(parent, bg=ModernTheme.CARD_BG)
        hop_header.pack(fill='x', padx=15, pady=(20, 10))
        
        ttk.Label(hop_header, text="🌐 Network Hops", 
                 style='Header.TLabel').pack(side='left')
        
        ttk.Button(hop_header, text="Configure Responses",
                  style='Modern.TButton',
                  command=self.configure_selected_hop).pack(side='right')
        
        # Hop list
        hop_list_frame = tk.Frame(parent, bg=ModernTheme.CARD_BG)
        hop_list_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        # Treeview for hops
        columns = ('Hop', 'IP', 'RTT', 'Location', 'Response')
        self.hop_tree = ttk.Treeview(hop_list_frame, columns=columns, 
                                    style='Modern.Treeview', height=12)
        self.hop_tree.pack(fill='both', expand=True)
        
        # Configure columns
        self.hop_tree.heading('#0', text='')
        self.hop_tree.column('#0', width=20)
        
        for col in columns:
            self.hop_tree.heading(col, text=col)
        
        self.hop_tree.column('Hop', width=50)
        self.hop_tree.column('IP', width=120)
        self.hop_tree.column('RTT', width=80)
        self.hop_tree.column('Location', width=100)
        self.hop_tree.column('Response', width=80)
        
        # Bind events
        self.hop_tree.bind('<Double-1>', self.on_hop_double_click)
        self.hop_tree.bind('<Button-3>', self.show_hop_context_menu)
        
        # Scrollbar
        hop_scrollbar = ttk.Scrollbar(hop_list_frame, orient=tk.VERTICAL, 
                                     command=self.hop_tree.yview)
        self.hop_tree.configure(yscrollcommand=hop_scrollbar.set)
        hop_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_network_map(self, parent):
        """Create network path visualization map"""
        # Map header
        map_header = tk.Frame(parent, bg=ModernTheme.DARK_BG)
        map_header.pack(fill='x', pady=(0, 15))
        
        ttk.Label(map_header, text="🗺️ Network Path Visualization", 
                 style='Header.TLabel').pack(side='left')
        
        # Map controls
        map_controls = tk.Frame(map_header, bg=ModernTheme.DARK_BG)
        map_controls.pack(side='right')
        
        ttk.Button(map_controls, text="Zoom In",
                  style='Modern.TButton',
                  command=self.zoom_in).pack(side='left', padx=(0, 5))
        ttk.Button(map_controls, text="Zoom Out",
                  style='Modern.TButton',
                  command=self.zoom_out).pack(side='left', padx=(0, 5))
        ttk.Button(map_controls, text="Reset View",
                  style='Modern.TButton',
                  command=self.reset_map_view).pack(side='left')
        
        # Map canvas
        map_frame = tk.Frame(parent, bg=ModernTheme.DARKER_BG, relief='solid', bd=1)
        map_frame.pack(fill='both', expand=True)
        
        self.map_canvas = tk.Canvas(map_frame, bg=ModernTheme.DARKER_BG, 
                                   highlightthickness=0)
        self.map_canvas.pack(fill='both', expand=True)
        
        # Bind canvas events
        self.map_canvas.bind('<Button-1>', self.on_map_click)
        self.map_canvas.bind('<B1-Motion>', self.on_map_drag)
        self.map_canvas.bind('<MouseWheel>', self.on_map_scroll)
        
        # Map state
        self.map_scale = 1.0
        self.map_offset_x = 0
        self.map_offset_y = 0
        self.map_objects = {}
        
        # Draw initial map
        self.draw_world_map()
    
    def draw_world_map(self):
        """Draw basic world map outline"""
        self.map_canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = self.map_canvas.winfo_width() or 800
        canvas_height = self.map_canvas.winfo_height() or 600
        
        # Draw grid
        self.draw_coordinate_grid(canvas_width, canvas_height)
        
        # Draw simple world outline (very basic)
        self.draw_continents(canvas_width, canvas_height)
        
        # Draw legend
        self.draw_map_legend(canvas_width, canvas_height)
        
        # Redraw hops if any exist
        self.redraw_hops()
    
    def draw_coordinate_grid(self, width, height):
        """Draw coordinate grid on map"""
        # Longitude lines (vertical)
        for lon in range(-180, 181, 30):
            x = self.lon_to_x(lon, width)
            self.map_canvas.create_line(x, 0, x, height, 
                                       fill=ModernTheme.BORDER_COLOR, 
                                       width=1, tags="grid")
        
        # Latitude lines (horizontal)  
        for lat in range(-90, 91, 30):
            y = self.lat_to_y(lat, height)
            self.map_canvas.create_line(0, y, width, y, 
                                       fill=ModernTheme.BORDER_COLOR, 
                                       width=1, tags="grid")
    
    def draw_continents(self, width, height):
        """Draw basic continent outlines"""
        # Very simplified continent boundaries
        continents = [
            # North America (simplified)
            [(-140, 60), (-100, 60), (-80, 40), (-120, 30), (-140, 50)],
            # Europe (simplified)
            [(-10, 60), (40, 60), (40, 40), (-10, 40)],
            # Asia (simplified)
            [(40, 70), (140, 70), (140, 20), (40, 20)],
            # Africa (simplified)
            [(-20, 35), (50, 35), (40, -35), (20, -35), (-20, 0)],
            # South America (simplified)
            [(-80, 10), (-40, 10), (-60, -55), (-80, -30)],
            # Australia (simplified)
            [(110, -10), (155, -10), (155, -45), (110, -45)]
        ]
        
        for continent in continents:
            points = []
            for lon, lat in continent:
                x = self.lon_to_x(lon, width)
                y = self.lat_to_y(lat, height)
                points.extend([x, y])
            
            if len(points) >= 6:  # At least 3 points
                self.map_canvas.create_polygon(points, 
                                             outline=ModernTheme.TEXT_SECONDARY,
                                             fill=ModernTheme.ACCENT_BG,
                                             width=1, tags="continent")
    
    def draw_map_legend(self, width, height):
        """Draw map legend"""
        legend_x = width - 150
        legend_y = 20
        
        # Legend background
        self.map_canvas.create_rectangle(legend_x, legend_y, legend_x + 140, legend_y + 100,
                                        fill=ModernTheme.CARD_BG, outline=ModernTheme.BORDER_COLOR,
                                        tags="legend")
        
        # Legend title
        self.map_canvas.create_text(legend_x + 70, legend_y + 15, text="Network Hops",
                                   fill=ModernTheme.TEXT_PRIMARY, font=('Segoe UI', 10, 'bold'),
                                   tags="legend")
        
        # Legend items
        legend_items = [
            ("🟢 Source", ModernTheme.SUCCESS_COLOR),
            ("🔵 Hop", ModernTheme.PRIMARY_COLOR),
            ("🔴 Target", ModernTheme.ERROR_COLOR),
            ("⚡ Selected", ModernTheme.WARNING_COLOR)
        ]
        
        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_y + 35 + (i * 15)
            self.map_canvas.create_oval(legend_x + 10, y_pos - 3, legend_x + 16, y_pos + 3,
                                       fill=color, outline="", tags="legend")
            self.map_canvas.create_text(legend_x + 25, y_pos, text=text.split(' ', 1)[1],
                                       fill=ModernTheme.TEXT_PRIMARY, font=('Segoe UI', 8),
                                       anchor='w', tags="legend")
    
    def lon_to_x(self, longitude, width):
        """Convert longitude to canvas x coordinate"""
        return int((longitude + 180) * width / 360 * self.map_scale + self.map_offset_x)
    
    def lat_to_y(self, latitude, height):
        """Convert latitude to canvas y coordinate"""
        return int((90 - latitude) * height / 180 * self.map_scale + self.map_offset_y)
    
    def x_to_lon(self, x, width):
        """Convert canvas x coordinate to longitude"""
        return (x - self.map_offset_x) / self.map_scale * 360 / width - 180
    
    def y_to_lat(self, y, height):
        """Convert canvas y coordinate to latitude"""
        return 90 - (y - self.map_offset_y) / self.map_scale * 180 / height
    
    def start_route_discovery(self):
        """Start route discovery process"""
        target = self.target_var.get().strip()
        if not target:
            messagebox.showerror("Error", "Please enter a target host")
            return
        
        # Validate target
        try:
            socket.gethostbyname(target)
        except socket.error:
            messagebox.showerror("Error", f"Cannot resolve target host: {target}")
            return
        
        # Show legal warning
        if not self.show_legal_warning():
            return
        
        self.target_host = target
        self.is_probing = True
        self.hops = []
        self.hop_locations = {}
        
        # Update UI
        self.start_probe_btn.config(state='disabled')
        self.stop_probe_btn.config(state='normal')
        self.probe_status.set_status('active')
        
        # Clear existing data
        for item in self.hop_tree.get_children():
            self.hop_tree.delete(item)
        
        # Start discovery in separate thread
        discovery_thread = threading.Thread(target=self.discovery_worker, 
                                           args=(target,), daemon=True)
        discovery_thread.start()
        
        self.add_log(f"Started route discovery to {target}", "INFO")
    
    def stop_route_discovery(self):
        """Stop route discovery process"""
        self.is_probing = False
        self.start_probe_btn.config(state='normal')
        self.stop_probe_btn.config(state='disabled')
        self.probe_status.set_status('inactive')
        
        self.add_log("Route discovery stopped", "INFO")
    
    def discovery_worker(self, target):
        """Worker thread for route discovery"""
        max_hops = int(self.max_hops_var.get())
        timeout = int(self.timeout_var.get())
        probe_type = self.probe_type_var.get()
        
        try:
            if probe_type == "ICMP" and SCAPY_AVAILABLE:
                self.icmp_traceroute(target, max_hops, timeout)
            else:
                # Fallback to system traceroute
                self.system_traceroute(target, max_hops)
                
        except Exception as e:
            self.add_log(f"Discovery error: {str(e)}", "ERROR")
        finally:
            # Reset UI state
            self.master.master.master.after(0, self.discovery_complete)
    
    def icmp_traceroute(self, target, max_hops, timeout):
        """Perform ICMP traceroute using Scapy"""
        for ttl in range(1, max_hops + 1):
            if not self.is_probing:
                break
                
            try:
                # Create ICMP packet with specific TTL
                packet = IP(dst=target, ttl=ttl) / ICMP()
                
                # Send packet and measure time
                start_time = time.time()
                reply = sr1(packet, timeout=timeout, verbose=False)
                rtt = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                if reply:
                    hop_ip = reply.src
                    hop_info = {
                        'hop': ttl,
                        'ip': hop_ip,
                        'rtt': f"{rtt:.2f} ms",
                        'location': 'Looking up...',
                        'response': 'Default'
                    }
                    
                    self.hops.append(hop_info)
                    
                    # Update UI in main thread
                    self.master.master.master.after(0, lambda: self.add_hop_to_tree(hop_info))
                    
                    # Get geolocation
                    self.get_hop_location(hop_ip, ttl)
                    
                    # Check if we reached the target
                    if hop_ip == socket.gethostbyname(target):
                        break
                        
                else:
                    # Timeout
                    hop_info = {
                        'hop': ttl,
                        'ip': '*',
                        'rtt': 'Timeout',
                        'location': 'Unknown',
                        'response': 'Timeout'
                    }
                    
                    self.hops.append(hop_info)
                    self.master.master.master.after(0, lambda: self.add_hop_to_tree(hop_info))
                
                # Small delay between probes
                time.sleep(0.1)
                
            except Exception as e:
                self.add_log(f"Hop {ttl} error: {str(e)}", "ERROR")
    
    def system_traceroute(self, target, max_hops):
        """Fallback system traceroute"""
        try:
            # Use system traceroute command
            if sys.platform.startswith('win'):
                cmd = ['tracert', '-h', str(max_hops), target]
            else:
                cmd = ['traceroute', '-m', str(max_hops), target]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, text=True)
            
            hop_num = 1
            for line in process.stdout:
                if not self.is_probing:
                    process.terminate()
                    break
                
                # Parse traceroute output (simplified)
                if any(char.isdigit() for char in line):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        # Extract IP if present
                        ip = 'Unknown'
                        rtt = 'Unknown'
                        
                        for part in parts:
                            if '.' in part and part.replace('.', '').replace(' ', '').isdigit():
                                ip = part
                            elif 'ms' in part:
                                rtt = part
                        
                        hop_info = {
                            'hop': hop_num,
                            'ip': ip,
                            'rtt': rtt,
                            'location': 'Looking up...',
                            'response': 'Default'
                        }
                        
                        self.hops.append(hop_info)
                        self.master.master.master.after(0, lambda: self.add_hop_to_tree(hop_info))
                        
                        if ip != 'Unknown':
                            self.get_hop_location(ip, hop_num)
                        
                        hop_num += 1
            
        except Exception as e:
            self.add_log(f"System traceroute error: {str(e)}", "ERROR")
    
    def get_hop_location(self, ip, hop_num):
        """Get geolocation for hop IP"""
        if ip == '*' or ip == 'Unknown' or not REQUESTS_AVAILABLE:
            return
        
        try:
            # Use free geolocation API
            response = requests.get(f"http://ip-api.com/json/{ip}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    location = f"{data.get('city', 'Unknown')}, {data.get('country', 'Unknown')}"
                    lat = data.get('lat', 0)
                    lon = data.get('lon', 0)
                    
                    self.hop_locations[ip] = {
                        'location': location,
                        'lat': lat,
                        'lon': lon,
                        'country': data.get('country', 'Unknown'),
                        'isp': data.get('isp', 'Unknown')
                    }
                    
                    # Update hop in tree
                    self.master.master.master.after(0, lambda: self.update_hop_location(hop_num, location))
                    
                    # Update map
                    self.master.master.master.after(0, lambda: self.add_hop_to_map(ip, hop_num, lat, lon))
                
        except Exception as e:
            self.add_log(f"Geolocation lookup failed for {ip}: {str(e)}", "WARNING")
    
    def add_hop_to_tree(self, hop_info):
        """Add hop to the tree view"""
        self.hop_tree.insert('', 'end', iid=f"hop_{hop_info['hop']}", 
                            values=(hop_info['hop'], hop_info['ip'], hop_info['rtt'], 
                                   hop_info['location'], hop_info['response']))
    
    def update_hop_location(self, hop_num, location):
        """Update hop location in tree view"""
        try:
            item_id = f"hop_{hop_num}"
            if self.hop_tree.exists(item_id):
                current_values = list(self.hop_tree.item(item_id)['values'])
                current_values[3] = location  # Update location column
                self.hop_tree.item(item_id, values=current_values)
        except Exception as e:
            pass  # Ignore update errors
    
    def add_hop_to_map(self, ip, hop_num, lat, lon):
        """Add hop to the network map"""
        if not self.map_canvas:
            return
        
        canvas_width = self.map_canvas.winfo_width() or 800
        canvas_height = self.map_canvas.winfo_height() or 600
        
        x = self.lon_to_x(lon, canvas_width)
        y = self.lat_to_y(lat, canvas_height)
        
        # Determine hop color
        if hop_num == 1:
            color = ModernTheme.SUCCESS_COLOR  # Source
        elif ip == socket.gethostbyname(self.target_host):
            color = ModernTheme.ERROR_COLOR  # Target
        else:
            color = ModernTheme.PRIMARY_COLOR  # Intermediate hop
        
        # Draw hop circle
        radius = 8
        hop_id = self.map_canvas.create_oval(x - radius, y - radius, 
                                           x + radius, y + radius,
                                           fill=color, outline=ModernTheme.TEXT_PRIMARY,
                                           width=2, tags=f"hop_{ip}")
        
        # Add hop number label
        self.map_canvas.create_text(x, y, text=str(hop_num),
                                   fill=ModernTheme.TEXT_PRIMARY,
                                   font=('Segoe UI', 8, 'bold'),
                                   tags=f"hop_{ip}")
        
        # Draw line to previous hop if exists
        if len(self.hops) > 1:
            prev_hop = self.hops[-2]
            if prev_hop['ip'] in self.hop_locations:
                prev_loc = self.hop_locations[prev_hop['ip']]
                prev_x = self.lon_to_x(prev_loc['lon'], canvas_width)
                prev_y = self.lat_to_y(prev_loc['lat'], canvas_height)
                
                self.map_canvas.create_line(prev_x, prev_y, x, y,
                                          fill=ModernTheme.PRIMARY_COLOR,
                                          width=2, tags="path")
        
        # Store hop object reference
        self.map_objects[ip] = hop_id
    
    def redraw_hops(self):
        """Redraw all hops on the map"""
        # Clear existing hop objects
        self.map_canvas.delete("hop")
        self.map_canvas.delete("path")
        
        canvas_width = self.map_canvas.winfo_width() or 800
        canvas_height = self.map_canvas.winfo_height() or 600
        
        # Redraw all hops
        for i, hop in enumerate(self.hops):
            if hop['ip'] in self.hop_locations:
                loc = self.hop_locations[hop['ip']]
                self.add_hop_to_map(hop['ip'], hop['hop'], loc['lat'], loc['lon'])
    
    def discovery_complete(self):
        """Handle discovery completion"""
        self.is_probing = False
        self.start_probe_btn.config(state='normal')
        self.stop_probe_btn.config(state='disabled')
        self.probe_status.set_status('inactive')
        
        self.add_log(f"Route discovery complete. Found {len(self.hops)} hops.", "SUCCESS")
    
    def on_map_click(self, event):
        """Handle map click events"""
        canvas_width = self.map_canvas.winfo_width()
        canvas_height = self.map_canvas.winfo_height()
        
        # Find clicked hop
        clicked_item = self.map_canvas.find_closest(event.x, event.y)[0]
        tags = self.map_canvas.gettags(clicked_item)
        
        for tag in tags:
            if tag.startswith('hop_'):
                hop_ip = tag[4:]  # Remove 'hop_' prefix
                self.select_hop_by_ip(hop_ip)
                self.show_hop_response_modal(hop_ip)
                break
    
    def select_hop_by_ip(self, ip):
        """Select hop by IP address"""
        # Find and select hop in tree
        for item_id in self.hop_tree.get_children():
            hop_values = self.hop_tree.item(item_id)['values']
            if hop_values[1] == ip:  # IP column
                self.hop_tree.selection_set(item_id)
                self.selected_hop = ip
                break
    
    def show_hop_response_modal(self, ip):
        """Show hop response configuration modal"""
        if ip not in self.hop_locations:
            return
        
        modal = tk.Toplevel(self.master.master.master)
        modal.title(f"Configure Hop Response - {ip}")
        modal.geometry("500x600")
        modal.configure(bg=ModernTheme.DARK_BG)
        modal.transient(self.master.master.master)
        modal.grab_set()
        
        # Center the modal
        modal.geometry("+%d+%d" % (self.master.master.master.winfo_rootx() + 50, 
                                  self.master.master.master.winfo_rooty() + 50))
        
        # Modal header
        header_frame = tk.Frame(modal, bg=ModernTheme.DARKER_BG)
        header_frame.pack(fill='x', pady=(0, 20))
        
        title_label = tk.Label(header_frame, text=f"🌐 Hop Configuration",
                              bg=ModernTheme.DARKER_BG,
                              fg=ModernTheme.TEXT_PRIMARY,
                              font=('Segoe UI', 14, 'bold'))
        title_label.pack(pady=15)
        
        # Hop information
        info_frame = tk.Frame(modal, bg=ModernTheme.CARD_BG, relief='solid', bd=1)
        info_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        hop_info = self.hop_locations[ip]
        
        info_items = [
            ("IP Address:", ip),
            ("Location:", hop_info['location']),
            ("Country:", hop_info['country']),
            ("ISP:", hop_info['isp']),
            ("Coordinates:", f"{hop_info['lat']:.4f}, {hop_info['lon']:.4f}")
        ]
        
        for label, value in info_items:
            item_frame = tk.Frame(info_frame, bg=ModernTheme.CARD_BG)
            item_frame.pack(fill='x', padx=15, pady=5)
            
            tk.Label(item_frame, text=label, bg=ModernTheme.CARD_BG,
                    fg=ModernTheme.TEXT_SECONDARY, font=('Segoe UI', 10)).pack(side='left')
            tk.Label(item_frame, text=value, bg=ModernTheme.CARD_BG,
                    fg=ModernTheme.TEXT_PRIMARY, font=('Segoe UI', 10, 'bold')).pack(side='right')
        
        # Response configuration
        response_frame = tk.Frame(modal, bg=ModernTheme.DARK_BG)
        response_frame.pack(fill='both', expand=True, padx=20)
        
        tk.Label(response_frame, text="Response Configuration:",
                bg=ModernTheme.DARK_BG, fg=ModernTheme.PRIMARY_COLOR,
                font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=(0, 10))
        
        # Response type selection
        response_type_frame = tk.Frame(response_frame, bg=ModernTheme.DARK_BG)
        response_type_frame.pack(fill='x', pady=10)
        
        tk.Label(response_type_frame, text="Response Type:",
                bg=ModernTheme.DARK_BG, fg=ModernTheme.TEXT_PRIMARY).pack(anchor='w')
        
        response_type_var = tk.StringVar(value=self.hop_responses.get(ip, {}).get('type', 'Default'))
        response_types = ["Default", "Drop", "Delay", "Modify", "Mirror", "Custom"]
        
        for response_type in response_types:
            rb = tk.Radiobutton(response_type_frame, text=response_type,
                               variable=response_type_var, value=response_type,
                               bg=ModernTheme.DARK_BG, fg=ModernTheme.TEXT_PRIMARY,
                               selectcolor=ModernTheme.ACCENT_BG, activebackground=ModernTheme.DARK_BG)
            rb.pack(anchor='w', padx=20)
        
        # Response parameters
        params_frame = tk.Frame(response_frame, bg=ModernTheme.ACCENT_BG, relief='solid', bd=1)
        params_frame.pack(fill='x', pady=10)
        
        tk.Label(params_frame, text="Parameters:",
                bg=ModernTheme.ACCENT_BG, fg=ModernTheme.TEXT_PRIMARY,
                font=('Segoe UI', 10, 'bold')).pack(anchor='w', padx=10, pady=(10, 5))
        
        # Delay parameter
        delay_frame = tk.Frame(params_frame, bg=ModernTheme.ACCENT_BG)
        delay_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(delay_frame, text="Delay (ms):", bg=ModernTheme.ACCENT_BG,
                fg=ModernTheme.TEXT_PRIMARY).pack(side='left')
        delay_var = tk.StringVar(value=self.hop_responses.get(ip, {}).get('delay', '0'))
        tk.Entry(delay_frame, textvariable=delay_var, width=10,
                bg=ModernTheme.DARKER_BG, fg=ModernTheme.TEXT_PRIMARY).pack(side='right')
        
        # Packet loss parameter
        loss_frame = tk.Frame(params_frame, bg=ModernTheme.ACCENT_BG)
        loss_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(loss_frame, text="Packet Loss (%):", bg=ModernTheme.ACCENT_BG,
                fg=ModernTheme.TEXT_PRIMARY).pack(side='left')
        loss_var = tk.StringVar(value=self.hop_responses.get(ip, {}).get('loss', '0'))
        tk.Entry(loss_frame, textvariable=loss_var, width=10,
                bg=ModernTheme.DARKER_BG, fg=ModernTheme.TEXT_PRIMARY).pack(side='right')
        
        # Custom response
        custom_frame = tk.Frame(params_frame, bg=ModernTheme.ACCENT_BG)
        custom_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(custom_frame, text="Custom Response:",
                bg=ModernTheme.ACCENT_BG, fg=ModernTheme.TEXT_PRIMARY).pack(anchor='w')
        
        custom_text = tk.Text(custom_frame, height=4, width=50,
                             bg=ModernTheme.DARKER_BG, fg=ModernTheme.TEXT_PRIMARY,
                             font=('Consolas', 9))
        custom_text.pack(fill='x', pady=(5, 10))
        custom_text.insert(tk.END, self.hop_responses.get(ip, {}).get('custom', ''))
        
        # Buttons
        button_frame = tk.Frame(modal, bg=ModernTheme.DARK_BG)
        button_frame.pack(fill='x', padx=20, pady=20)
        
        def save_response():
            self.hop_responses[ip] = {
                'type': response_type_var.get(),
                'delay': delay_var.get(),
                'loss': loss_var.get(),
                'custom': custom_text.get(1.0, tk.END).strip()
            }
            
            # Update hop in tree
            self.update_hop_response(ip, response_type_var.get())
            
            self.add_log(f"Updated response configuration for hop {ip}", "INFO")
            modal.destroy()
        
        def test_response():
            # Simulate testing the response configuration
            self.add_log(f"Testing response configuration for hop {ip}", "INFO")
            messagebox.showinfo("Test", f"Response test initiated for {ip}")
        
        tk.Button(button_frame, text="Save Configuration",
                 bg=ModernTheme.SUCCESS_COLOR, fg=ModernTheme.TEXT_PRIMARY,
                 font=('Segoe UI', 10, 'bold'), command=save_response).pack(side='right', padx=(5, 0))
        
        tk.Button(button_frame, text="Test Response",
                 bg=ModernTheme.WARNING_COLOR, fg=ModernTheme.TEXT_PRIMARY,
                 font=('Segoe UI', 10, 'bold'), command=test_response).pack(side='right', padx=(5, 0))
        
        tk.Button(button_frame, text="Cancel",
                 bg=ModernTheme.ACCENT_BG, fg=ModernTheme.TEXT_PRIMARY,
                 font=('Segoe UI', 10), command=modal.destroy).pack(side='right')
    
    def update_hop_response(self, ip, response_type):
        """Update hop response type in tree view"""
        for item_id in self.hop_tree.get_children():
            hop_values = list(self.hop_tree.item(item_id)['values'])
            if hop_values[1] == ip:  # IP column
                hop_values[4] = response_type  # Response column
                self.hop_tree.item(item_id, values=hop_values)
                break
    
    def on_hop_double_click(self, event):
        """Handle double-click on hop in tree"""
        selection = self.hop_tree.selection()
        if selection:
            hop_values = self.hop_tree.item(selection[0])['values']
            hop_ip = hop_values[1]
            if hop_ip != '*':
                self.show_hop_response_modal(hop_ip)
    
    def configure_selected_hop(self):
        """Configure response for selected hop"""
        selection = self.hop_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a hop to configure")
            return
        
        hop_values = self.hop_tree.item(selection[0])['values']
        hop_ip = hop_values[1]
        
        if hop_ip == '*':
            messagebox.showwarning("Warning", "Cannot configure response for timeout hop")
            return
        
        self.show_hop_response_modal(hop_ip)
    
    def show_hop_context_menu(self, event):
        """Show context menu for hop"""
        # TODO: Implement context menu
        pass
    
    def zoom_in(self):
        """Zoom in on map"""
        self.map_scale *= 1.2
        self.draw_world_map()
    
    def zoom_out(self):
        """Zoom out on map"""
        self.map_scale /= 1.2
        self.draw_world_map()
    
    def reset_map_view(self):
        """Reset map view to default"""
        self.map_scale = 1.0
        self.map_offset_x = 0
        self.map_offset_y = 0
        self.draw_world_map()
    
    def on_map_drag(self, event):
        """Handle map dragging"""
        # TODO: Implement map dragging
        pass
    
    def on_map_scroll(self, event):
        """Handle map scrolling"""
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def clear_route(self):
        """Clear current route data"""
        self.hops = []
        self.hop_locations = {}
        self.hop_responses = {}
        
        # Clear tree
        for item in self.hop_tree.get_children():
            self.hop_tree.delete(item)
        
        # Clear map
        self.map_canvas.delete("hop")
        self.map_canvas.delete("path")
        
        self.add_log("Route data cleared", "INFO")
    
    def export_route(self):
        """Export route data to file"""
        if not self.hops:
            messagebox.showwarning("Warning", "No route data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Route Data",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                export_data = {
                    'target': self.target_host,
                    'timestamp': datetime.now().isoformat(),
                    'hops': self.hops,
                    'locations': self.hop_locations,
                    'responses': self.hop_responses
                }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                self.add_log(f"Route data exported to {filename}", "SUCCESS")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export route data: {e}")
    
    def import_route(self):
        """Import route data from file"""
        filename = filedialog.askopenfilename(
            title="Import Route Data",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    import_data = json.load(f)
                
                self.target_host = import_data.get('target', '')
                self.hops = import_data.get('hops', [])
                self.hop_locations = import_data.get('locations', {})
                self.hop_responses = import_data.get('responses', {})
                
                # Populate tree
                for item in self.hop_tree.get_children():
                    self.hop_tree.delete(item)
                
                for hop in self.hops:
                    self.add_hop_to_tree(hop)
                
                # Redraw map
                self.redraw_hops()
                
                self.add_log(f"Route data imported from {filename}", "SUCCESS")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import route data: {e}")
    
    def show_legal_warning(self) -> bool:
        """Show legal warning for network probing"""
        warning_text = """
⚠️ NETWORK PROBING AUTHORIZATION REQUIRED ⚠️

This tool will perform network route discovery (traceroute) to map
the path to the target host and gather geolocation information.

You MUST have proper authorization to:
• Probe network paths to the target
• Gather routing information
• Test network connectivity
• Configure response behaviors

UNAUTHORIZED NETWORK PROBING may violate:
• Network security policies
• Terms of service agreements
• Local and international laws

Do you have proper authorization for this network discovery?
        """
        
        return messagebox.askyesno("⚠️ Network Probing Authorization", warning_text)
    
    def add_log(self, message: str, level: str = "INFO"):
        """Add log entry (delegate to main app)"""
        if hasattr(self.master.master.master, 'add_log'):
            self.master.master.master.add_log(message, level)


class RoutingProtocolManager:
    """Advanced routing protocol management"""
    
    def __init__(self, parent_gui):
        self.parent_gui = parent_gui
        self.cisco_scripts = self.load_cisco_scripts()
        self.juniper_scripts = self.load_juniper_scripts()
        self.active_sessions = {}
    
    def load_cisco_scripts(self):
        """Load Cisco routing scripts"""
        return {
            "BGP Configuration": {
                "description": "Configure BGP routing with neighbor relationships",
                "script": """
! BGP Configuration Script
router bgp {as_number}
 bgp router-id {router_id}
 neighbor {neighbor_ip} remote-as {neighbor_as}
 neighbor {neighbor_ip} update-source {source_interface}
 address-family ipv4
  network {network} mask {netmask}
  neighbor {neighbor_ip} activate
 exit-address-family
!
""",
                "parameters": ["as_number", "router_id", "neighbor_ip", "neighbor_as", "source_interface", "network", "netmask"]
            },
            "OSPF Configuration": {
                "description": "Configure OSPF routing protocol",
                "script": """
! OSPF Configuration Script
router ospf {process_id}
 router-id {router_id}
 network {network} {wildcard} area {area}
 passive-interface {passive_interface}
 default-information originate
!
interface {interface}
 ip ospf hello-interval {hello_interval}
 ip ospf dead-interval {dead_interval}
!
""",
                "parameters": ["process_id", "router_id", "network", "wildcard", "area", "passive_interface", "interface", "hello_interval", "dead_interval"]
            },
            "EIGRP Configuration": {
                "description": "Configure EIGRP routing protocol",
                "script": """
! EIGRP Configuration Script
router eigrp {as_number}
 network {network} {wildcard}
 passive-interface {passive_interface}
 eigrp router-id {router_id}
!
interface {interface}
 ip hello-interval eigrp {as_number} {hello_interval}
 ip hold-time eigrp {as_number} {hold_time}
!
""",
                "parameters": ["as_number", "network", "wildcard", "passive_interface", "router_id", "interface", "hello_interval", "hold_time"]
            },
            "Static Route Configuration": {
                "description": "Configure static routes",
                "script": """
! Static Route Configuration
ip route {destination} {netmask} {next_hop} {admin_distance}
ip route {destination} {netmask} {interface}
!
""",
                "parameters": ["destination", "netmask", "next_hop", "admin_distance", "interface"]
            },
            "Route Redistribution": {
                "description": "Configure route redistribution between protocols",
                "script": """
! Route Redistribution Configuration
router {protocol1}
 redistribute {protocol2} metric {metric} subnets
!
router {protocol2}
 redistribute {protocol1} metric {metric}
!
""",
                "parameters": ["protocol1", "protocol2", "metric"]
            }
        }
    
    def load_juniper_scripts(self):
        """Load Juniper routing scripts"""
        return {
            "BGP Configuration": {
                "description": "Configure BGP routing protocol",
                "script": """
# Juniper BGP Configuration
set routing-options router-id {router_id}
set routing-options autonomous-system {as_number}
set protocols bgp group {group_name} type {type}
set protocols bgp group {group_name} peer-as {peer_as}
set protocols bgp group {group_name} neighbor {neighbor_ip}
set policy-options policy-statement {policy_name} then accept
set protocols bgp group {group_name} export {policy_name}
""",
                "parameters": ["router_id", "as_number", "group_name", "type", "peer_as", "neighbor_ip", "policy_name"]
            },
            "OSPF Configuration": {
                "description": "Configure OSPF routing protocol",
                "script": """
# Juniper OSPF Configuration
set routing-options router-id {router_id}
set protocols ospf area {area} interface {interface}
set protocols ospf area {area} interface {interface} hello-interval {hello_interval}
set protocols ospf area {area} interface {interface} dead-interval {dead_interval}
set protocols ospf area {area} interface {interface} priority {priority}
""",
                "parameters": ["router_id", "area", "interface", "hello_interval", "dead_interval", "priority"]
            },
            "IS-IS Configuration": {
                "description": "Configure IS-IS routing protocol",
                "script": """
# Juniper IS-IS Configuration
set protocols isis interface {interface}
set protocols isis level {level} wide-metrics-only
set interfaces {interface} unit 0 family iso address {iso_address}
set protocols isis area-password {password}
""",
                "parameters": ["interface", "level", "iso_address", "password"]
            },
            "Static Route Configuration": {
                "description": "Configure static routes",
                "script": """
# Juniper Static Route Configuration
set routing-options static route {destination}/{prefix} next-hop {next_hop}
set routing-options static route {destination}/{prefix} preference {preference}
""",
                "parameters": ["destination", "prefix", "next_hop", "preference"]
            },
            "RIP Configuration": {
                "description": "Configure RIP routing protocol",
                "script": """
# Juniper RIP Configuration
set protocols rip group {group_name} neighbor {neighbor}
set protocols rip group {group_name} export {export_policy}
set protocols rip group {group_name} import {import_policy}
""",
                "parameters": ["group_name", "neighbor", "export_policy", "import_policy"]
            }
        }
    
    def execute_cisco_script(self, script_name, parameters, target_device):
        """Execute Cisco script on target device"""
        if not NETMIKO_AVAILABLE:
            raise Exception("Netmiko library not available for device connection")
        
        script_data = self.cisco_scripts.get(script_name)
        if not script_data:
            raise Exception(f"Script '{script_name}' not found")
        
        # Format script with parameters
        try:
            formatted_script = script_data["script"].format(**parameters)
        except KeyError as e:
            raise Exception(f"Missing parameter: {e}")
        
        # Connect to device and execute
        device = {
            'device_type': 'cisco_ios',
            'host': target_device['host'],
            'username': target_device['username'],
            'password': target_device['password'],
            'port': target_device.get('port', 22),
        }
        
        try:
            connection = netmiko.ConnectHandler(**device)
            connection.enable()
            
            # Send configuration commands
            output = connection.send_config_set(formatted_script.split('\n'))
            connection.save_config()
            connection.disconnect()
            
            return output
        except Exception as e:
            raise Exception(f"Failed to execute script on device: {e}")
    
    def execute_juniper_script(self, script_name, parameters, target_device):
        """Execute Juniper script on target device"""
        if not NETMIKO_AVAILABLE:
            raise Exception("Netmiko library not available for device connection")
        
        script_data = self.juniper_scripts.get(script_name)
        if not script_data:
            raise Exception(f"Script '{script_name}' not found")
        
        # Format script with parameters
        try:
            formatted_script = script_data["script"].format(**parameters)
        except KeyError as e:
            raise Exception(f"Missing parameter: {e}")
        
        # Connect to device and execute
        device = {
            'device_type': 'juniper',
            'host': target_device['host'],
            'username': target_device['username'],
            'password': target_device['password'],
            'port': target_device.get('port', 22),
        }
        
        try:
            connection = netmiko.ConnectHandler(**device)
            
            # Enter configuration mode
            connection.config_mode()
            
            # Send configuration commands
            output = connection.send_config_set(formatted_script.split('\n'))
            
            # Commit configuration
            connection.commit()
            connection.exit_config_mode()
            connection.disconnect()
            
            return output
        except Exception as e:
            raise Exception(f"Failed to execute script on device: {e}")


class AdvancedNetworkTestingGUI:
    """Advanced Network Protocol Testing Suite GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Network Protocol Testing Suite")
        self.root.geometry("1600x1000")
        self.root.configure(bg=ModernTheme.DARK_BG)
        
        # Configure modern theme
        ModernTheme.configure_style()
        
        # Application state
        self.is_generating = False
        self.current_config = {}
        self.stats = {
            'packets_sent': 0,
            'bytes_sent': 0,
            'start_time': None,
            'protocol_stats': {}
        }
        
        # Message queue for thread communication
        self.message_queue = queue.Queue()
        
        # Initialize routing protocol manager
        self.routing_manager = RoutingProtocolManager(self)
        
        # Create GUI
        self.create_header()
        self.create_main_interface()
        self.create_status_bar()
        
        # Start message processing
        self.process_messages()
        
        # Check dependencies
        self.check_dependencies()
    
    def create_header(self):
        """Create professional application header"""
        header_frame = tk.Frame(self.root, bg=ModernTheme.DARKER_BG, height=70)
        header_frame.pack(fill='x', pady=(0, 1))
        header_frame.pack_propagate(False)
        
        # Logo/Title section
        title_frame = tk.Frame(header_frame, bg=ModernTheme.DARKER_BG)
        title_frame.pack(side='left', fill='y', padx=25)
        
        # Application icon (network symbol)
        icon_label = tk.Label(title_frame, text="🌐", 
                             bg=ModernTheme.DARKER_BG, 
                             fg=ModernTheme.PRIMARY_COLOR,
                             font=('Segoe UI', 24))
        icon_label.pack(side='left', pady=18)
        
        # Title and subtitle
        title_text_frame = tk.Frame(title_frame, bg=ModernTheme.DARKER_BG)
        title_text_frame.pack(side='left', padx=(15, 0), pady=12)
        
        title_label = tk.Label(title_text_frame, text="Advanced Network Protocol Testing Suite",
                              bg=ModernTheme.DARKER_BG,
                              fg=ModernTheme.TEXT_PRIMARY,
                              font=('Segoe UI', 16, 'bold'))
        title_label.pack(anchor='w')
        
        subtitle_label = tk.Label(title_text_frame, text="Professional Router Testing • Packet Crafting • Protocol Simulation",
                                 bg=ModernTheme.DARKER_BG,
                                 fg=ModernTheme.TEXT_SECONDARY,
                                 font=('Segoe UI', 10))
        subtitle_label.pack(anchor='w')
        
        # Status indicators section
        status_frame = tk.Frame(header_frame, bg=ModernTheme.DARKER_BG)
        status_frame.pack(side='right', fill='y', padx=25)
        
        # Create status grid
        indicators_frame = tk.Frame(status_frame, bg=ModernTheme.DARKER_BG)
        indicators_frame.pack(side='right', pady=12)
        
        # First row of indicators
        row1_frame = tk.Frame(indicators_frame, bg=ModernTheme.DARKER_BG)
        row1_frame.pack(anchor='e', pady=2)
        
        self.scapy_status = StatusIndicator(row1_frame, "Scapy Engine")
        self.scapy_status.pack(side='left', padx=(0, 15))
        
        self.netmiko_status = StatusIndicator(row1_frame, "Device Access")
        self.netmiko_status.pack(side='left')
        
        # Second row of indicators
        row2_frame = tk.Frame(indicators_frame, bg=ModernTheme.DARKER_BG)
        row2_frame.pack(anchor='e', pady=2)
        
        self.network_status = StatusIndicator(row2_frame, "Network")
        self.network_status.pack(side='left', padx=(0, 15))
        
        self.routing_status = StatusIndicator(row2_frame, "Routing Engine")
        self.routing_status.pack(side='left')
        
        # Update initial status
        self.scapy_status.set_status('active' if SCAPY_AVAILABLE else 'inactive')
        self.netmiko_status.set_status('active' if NETMIKO_AVAILABLE else 'inactive')
        self.network_status.set_status('active')
        self.routing_status.set_status('active')
    
    def create_main_interface(self):
        """Create main tabbed interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg=ModernTheme.DARK_BG)
        main_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame, style='Modern.TNotebook')
        self.notebook.pack(fill='both', expand=True)
        
        # Create all tabs
        self.create_dashboard_tab()
        self.create_packet_crafter_tab()
        self.create_network_path_tab()
        self.create_routing_protocols_tab()
        self.create_cisco_scripts_tab()
        self.create_juniper_scripts_tab()
        self.create_monitoring_tab()
        self.create_configuration_tab()
        self.create_logs_tab()
    
    def create_dashboard_tab(self):
        """Create enhanced dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(dashboard_frame, text="🏠 Dashboard")
        
        # Top metrics section
        metrics_section = tk.Frame(dashboard_frame, bg=ModernTheme.DARK_BG, height=120)
        metrics_section.pack(fill='x', padx=20, pady=20)
        metrics_section.pack_propagate(False)
        
        # Metrics cards
        self.packets_card = MetricCard(metrics_section, "Packets Sent", "0", "pkt", "📤")
        self.packets_card.pack(side='left', padx=(0, 15), fill='y')
        
        self.bytes_card = MetricCard(metrics_section, "Data Transmitted", "0", "B", "📊")
        self.bytes_card.pack(side='left', padx=(0, 15), fill='y')
        
        self.rate_card = MetricCard(metrics_section, "Current Rate", "0", "pps", "⚡")
        self.rate_card.pack(side='left', padx=(0, 15), fill='y')
        
        self.sessions_card = MetricCard(metrics_section, "Active Sessions", "0", "", "🔗")
        self.sessions_card.pack(side='left', fill='y')
        
        # Main content area
        content_frame = tk.Frame(dashboard_frame, bg=ModernTheme.DARK_BG)
        content_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Left panel - Quick Actions
        left_panel = tk.Frame(content_frame, bg=ModernTheme.CARD_BG, width=350)
        left_panel.pack(side='left', fill='y', padx=(0, 15))
        left_panel.pack_propagate(False)
        
        # Quick Actions header
        quick_header = tk.Frame(left_panel, bg=ModernTheme.CARD_BG)
        quick_header.pack(fill='x', padx=20, pady=20)
        
        ttk.Label(quick_header, text="⚡ Quick Actions", 
                 style='Header.TLabel').pack(side='left')
        
        # Protocol selector
        protocol_frame = tk.Frame(left_panel, bg=ModernTheme.CARD_BG)
        protocol_frame.pack(fill='x', padx=20, pady=(0, 15))
        
        ttk.Label(protocol_frame, text="Protocol:", style='Modern.TLabel').pack(anchor='w', pady=(0, 5))
        self.quick_protocol_var = tk.StringVar(value="ICMP")
        protocol_combo = ttk.Combobox(protocol_frame, textvariable=self.quick_protocol_var,
                                     values=["ICMP", "TCP", "UDP", "BGP", "OSPF", "EIGRP", "MQTT", "SNMP"],
                                     style='Modern.TCombobox', state='readonly')
        protocol_combo.pack(fill='x')
        
        # Target input
        target_frame = tk.Frame(left_panel, bg=ModernTheme.CARD_BG)
        target_frame.pack(fill='x', padx=20, pady=(0, 15))
        
        ttk.Label(target_frame, text="Target:", style='Modern.TLabel').pack(anchor='w', pady=(0, 5))
        self.quick_target_var = tk.StringVar(value="192.168.1.1")
        target_entry = ttk.Entry(target_frame, textvariable=self.quick_target_var,
                                style='Modern.TEntry')
        target_entry.pack(fill='x')
        
        # Rate control
        rate_frame = tk.Frame(left_panel, bg=ModernTheme.CARD_BG)
        rate_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        ttk.Label(rate_frame, text="Rate (pps):", style='Modern.TLabel').pack(anchor='w', pady=(0, 5))
        self.quick_rate_var = tk.StringVar(value="1")
        rate_entry = ttk.Entry(rate_frame, textvariable=self.quick_rate_var,
                              style='Modern.TEntry')
        rate_entry.pack(fill='x')
        
        # Control buttons
        button_frame = tk.Frame(left_panel, bg=ModernTheme.CARD_BG)
        button_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        self.start_button = ttk.Button(button_frame, text="🚀 Start Generation",
                                      style='Success.TButton',
                                      command=self.start_traffic_generation)
        self.start_button.pack(fill='x', pady=(0, 8))
        
        self.stop_button = ttk.Button(button_frame, text="⏹ Stop Generation",
                                     style='Error.TButton',
                                     command=self.stop_traffic_generation,
                                     state='disabled')
        self.stop_button.pack(fill='x')
        
        # Right panel - Live Charts
        right_panel = tk.Frame(content_frame, bg=ModernTheme.DARK_BG)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Chart header
        chart_header = tk.Frame(right_panel, bg=ModernTheme.DARK_BG)
        chart_header.pack(fill='x', pady=(0, 15))
        
        ttk.Label(chart_header, text="📈 Real-time Traffic Analysis", 
                 style='Header.TLabel').pack(side='left')
        
        # Chart placeholder (would contain matplotlib charts)
        if MATPLOTLIB_AVAILABLE:
            chart_frame = tk.Frame(right_panel, bg=ModernTheme.DARKER_BG, relief='solid', bd=1)
            chart_frame.pack(fill='both', expand=True)
            
            chart_label = tk.Label(chart_frame, text="📊\n\nReal-time Traffic Charts\nWould display here with matplotlib",
                                  bg=ModernTheme.DARKER_BG,
                                  fg=ModernTheme.TEXT_SECONDARY,
                                  font=('Segoe UI', 12))
            chart_label.pack(expand=True)
        else:
            chart_frame = tk.Frame(right_panel, bg=ModernTheme.DARKER_BG, relief='solid', bd=1)
            chart_frame.pack(fill='both', expand=True)
            
            chart_label = tk.Label(chart_frame, text="📊\n\nMatplotlib not available\nInstall with: pip install matplotlib",
                                  bg=ModernTheme.DARKER_BG,
                                  fg=ModernTheme.WARNING_COLOR,
                                  font=('Segoe UI', 12))
            chart_label.pack(expand=True)
    
    def create_packet_crafter_tab(self):
        """Create packet crafting tab"""
        packet_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(packet_frame, text="🔧 Packet Crafter")
        
        # Create packet crafting widget
        self.packet_crafter = PacketCraftingWidget(packet_frame)
        self.packet_crafter.pack(fill='both', expand=True, padx=20, pady=20)
    
    def create_network_path_tab(self):
        """Create network path discovery tab"""
        path_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(path_frame, text="🗺️ Network Path")
        
        # Create network path discovery widget
        self.network_path = NetworkPathDiscovery(path_frame)
        self.network_path.pack(fill='both', expand=True, padx=20, pady=20)
    
    def create_routing_protocols_tab(self):
        """Create routing protocols tab"""
        routing_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(routing_frame, text="🌐 Routing Protocols")
        
        # Main content
        content_frame = tk.Frame(routing_frame, bg=ModernTheme.DARK_BG)
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(content_frame, bg=ModernTheme.DARK_BG)
        header_frame.pack(fill='x', pady=(0, 20))
        
        ttk.Label(header_frame, text="🌐 Routing Protocol Simulation", 
                 style='Title.TLabel').pack(side='left')
        
        # Protocol selection
        protocol_section = tk.Frame(content_frame, bg=ModernTheme.DARK_BG)
        protocol_section.pack(fill='x', pady=(0, 20))
        
        ttk.Label(protocol_section, text="Select Routing Protocol:", 
                 style='Header.TLabel').pack(anchor='w', pady=(0, 10))
        
        # Protocol buttons
        protocol_buttons_frame = tk.Frame(protocol_section, bg=ModernTheme.DARK_BG)
        protocol_buttons_frame.pack(fill='x')
        
        protocols = [
            ("BGP", "🔄 Border Gateway Protocol", self.simulate_bgp),
            ("OSPF", "🕷 Open Shortest Path First", self.simulate_ospf),
            ("EIGRP", "⚡ Enhanced Interior Gateway Routing", self.simulate_eigrp),
            ("IS-IS", "🔗 Intermediate System to Intermediate System", self.simulate_isis),
            ("RIP", "📍 Routing Information Protocol", self.simulate_rip)
        ]
        
        for i, (protocol, description, command) in enumerate(protocols):
            btn_frame = tk.Frame(protocol_buttons_frame, bg=ModernTheme.CARD_BG, relief='solid', bd=1)
            btn_frame.pack(fill='x', pady=5)
            
            btn = tk.Button(btn_frame, text=f"{description}",
                           bg=ModernTheme.CARD_BG,
                           fg=ModernTheme.TEXT_PRIMARY,
                           font=('Segoe UI', 11, 'bold'),
                           relief='flat',
                           cursor='hand2',
                           command=command)
            btn.pack(fill='x', padx=15, pady=15)
            
            # Hover effects
            def on_enter(e, button=btn):
                button.config(bg=ModernTheme.PRIMARY_COLOR)
            def on_leave(e, button=btn):
                button.config(bg=ModernTheme.CARD_BG)
            
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
        
        # Configuration area
        config_section = tk.Frame(content_frame, bg=ModernTheme.DARK_BG)
        config_section.pack(fill='both', expand=True)
        
        ttk.Label(config_section, text="Protocol Configuration:", 
                 style='Header.TLabel').pack(anchor='w', pady=(20, 10))
        
        self.routing_config_frame = tk.Frame(config_section, bg=ModernTheme.DARKER_BG, relief='solid', bd=1)
        self.routing_config_frame.pack(fill='both', expand=True)
        
        # Initial message
        initial_label = tk.Label(self.routing_config_frame, 
                                text="Select a routing protocol above to configure and simulate",
                                bg=ModernTheme.DARKER_BG,
                                fg=ModernTheme.TEXT_SECONDARY,
                                font=('Segoe UI', 14))
        initial_label.pack(expand=True)
    
    def create_cisco_scripts_tab(self):
        """Create Cisco scripts tab"""
        cisco_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(cisco_frame, text="🔵 Cisco Scripts")
        
        # Create Cisco script interface
        self.create_vendor_script_interface(cisco_frame, "Cisco", self.routing_manager.cisco_scripts, 
                                          self.execute_cisco_script, ModernTheme.CISCO_BLUE)
    
    def create_juniper_scripts_tab(self):
        """Create Juniper scripts tab"""
        juniper_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(juniper_frame, text="🟢 Juniper Scripts")
        
        # Create Juniper script interface
        self.create_vendor_script_interface(juniper_frame, "Juniper", self.routing_manager.juniper_scripts,
                                          self.execute_juniper_script, ModernTheme.JUNIPER_GREEN)
    
    def create_vendor_script_interface(self, parent, vendor, scripts, execute_function, brand_color):
        """Create vendor-specific script interface"""
        content_frame = tk.Frame(parent, bg=ModernTheme.DARK_BG)
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header with vendor branding
        header_frame = tk.Frame(content_frame, bg=ModernTheme.DARK_BG)
        header_frame.pack(fill='x', pady=(0, 20))
        
        vendor_icon = "🔵" if vendor == "Cisco" else "🟢"
        title_label = tk.Label(header_frame, text=f"{vendor_icon} {vendor} Network Configuration Scripts",
                              bg=ModernTheme.DARK_BG,
                              fg=brand_color,
                              font=('Segoe UI', 16, 'bold'))
        title_label.pack(side='left')
        
        # Connection status
        connection_status = StatusIndicator(header_frame, f"{vendor} Connection")
        connection_status.pack(side='right')
        
        # Main content area
        main_content = tk.Frame(content_frame, bg=ModernTheme.DARK_BG)
        main_content.pack(fill='both', expand=True)
        
        # Left panel - Script selection
        left_panel = tk.Frame(main_content, bg=ModernTheme.CARD_BG, width=400)
        left_panel.pack(side='left', fill='y', padx=(0, 15))
        left_panel.pack_propagate(False)
        
        # Device connection section
        device_section = tk.Frame(left_panel, bg=ModernTheme.CARD_BG)
        device_section.pack(fill='x', padx=15, pady=15)
        
        ttk.Label(device_section, text="Device Connection", 
                 style='SubHeader.TLabel').pack(anchor='w', pady=(0, 10))
        
        # Device connection fields
        self.create_device_connection_fields(device_section, vendor.lower())
        
        # Script selection section
        script_section = tk.Frame(left_panel, bg=ModernTheme.CARD_BG)
        script_section.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        ttk.Label(script_section, text="Available Scripts", 
                 style='SubHeader.TLabel').pack(anchor='w', pady=(0, 10))
        
        # Script list
        script_listbox = tk.Listbox(script_section,
                                   bg=ModernTheme.ACCENT_BG,
                                   fg=ModernTheme.TEXT_PRIMARY,
                                   selectbackground=brand_color,
                                   borderwidth=0,
                                   highlightthickness=0,
                                   font=('Segoe UI', 10))
        script_listbox.pack(fill='both', expand=True, pady=(0, 10))
        
        for script_name in scripts.keys():
            script_listbox.insert(tk.END, script_name)
        
        script_listbox.bind('<<ListboxSelect>>', 
                           lambda e: self.on_script_select(e, vendor, scripts))
        
        # Execute button
        execute_btn = tk.Button(script_section, 
                               text=f"Execute on {vendor} Device",
                               bg=brand_color,
                               fg=ModernTheme.TEXT_PRIMARY,
                               font=('Segoe UI', 11, 'bold'),
                               relief='flat',
                               cursor='hand2',
                               command=lambda: execute_function())
        execute_btn.pack(fill='x')
        
        # Right panel - Script preview and parameters
        right_panel = tk.Frame(main_content, bg=ModernTheme.DARK_BG)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Script preview
        preview_label = ttk.Label(right_panel, text="Script Preview", 
                                 style='Header.TLabel')
        preview_label.pack(anchor='w', pady=(0, 10))
        
        script_preview = scrolledtext.ScrolledText(right_panel,
                                                  bg=ModernTheme.DARKER_BG,
                                                  fg=ModernTheme.TEXT_PRIMARY,
                                                  font=('Consolas', 10),
                                                  height=12)
        script_preview.pack(fill='x', pady=(0, 20))
        
        # Parameters section
        params_label = ttk.Label(right_panel, text="Script Parameters", 
                                style='Header.TLabel')
        params_label.pack(anchor='w', pady=(0, 10))
        
        params_frame = tk.Frame(right_panel, bg=ModernTheme.DARKER_BG, relief='solid', bd=1)
        params_frame.pack(fill='both', expand=True)
        
        # Store references
        setattr(self, f"{vendor.lower()}_script_listbox", script_listbox)
        setattr(self, f"{vendor.lower()}_script_preview", script_preview)
        setattr(self, f"{vendor.lower()}_params_frame", params_frame)
    
    def create_device_connection_fields(self, parent, vendor):
        """Create device connection fields"""
        # Host
        host_frame = tk.Frame(parent, bg=ModernTheme.CARD_BG)
        host_frame.pack(fill='x', pady=3)
        
        ttk.Label(host_frame, text="Host:", style='Modern.TLabel').pack(side='left')
        host_var = tk.StringVar(value="192.168.1.1")
        ttk.Entry(host_frame, textvariable=host_var, 
                 style='Modern.TEntry', width=15).pack(side='right')
        setattr(self, f"{vendor}_host_var", host_var)
        
        # Username
        user_frame = tk.Frame(parent, bg=ModernTheme.CARD_BG)
        user_frame.pack(fill='x', pady=3)
        
        ttk.Label(user_frame, text="Username:", style='Modern.TLabel').pack(side='left')
        user_var = tk.StringVar(value="admin")
        ttk.Entry(user_frame, textvariable=user_var, 
                 style='Modern.TEntry', width=15).pack(side='right')
        setattr(self, f"{vendor}_user_var", user_var)
        
        # Password
        pass_frame = tk.Frame(parent, bg=ModernTheme.CARD_BG)
        pass_frame.pack(fill='x', pady=3)
        
        ttk.Label(pass_frame, text="Password:", style='Modern.TLabel').pack(side='left')
        pass_var = tk.StringVar()
        ttk.Entry(pass_frame, textvariable=pass_var, 
                 style='Modern.TEntry', width=15, show="*").pack(side='right')
        setattr(self, f"{vendor}_pass_var", pass_var)
        
        # Port
        port_frame = tk.Frame(parent, bg=ModernTheme.CARD_BG)
        port_frame.pack(fill='x', pady=3)
        
        ttk.Label(port_frame, text="Port:", style='Modern.TLabel').pack(side='left')
        port_var = tk.StringVar(value="22")
        ttk.Entry(port_frame, textvariable=port_var, 
                 style='Modern.TEntry', width=15).pack(side='right')
        setattr(self, f"{vendor}_port_var", port_var)
    
    def create_monitoring_tab(self):
        """Create network monitoring tab"""
        monitoring_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(monitoring_frame, text="📊 Monitoring")
        
        content_frame = tk.Frame(monitoring_frame, bg=ModernTheme.DARK_BG)
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header_label = ttk.Label(content_frame, text="📊 Network Monitoring & Analysis", 
                                style='Title.TLabel')
        header_label.pack(anchor='w', pady=(0, 20))
        
        # Monitoring interface placeholder
        monitoring_placeholder = tk.Frame(content_frame, bg=ModernTheme.DARKER_BG, relief='solid', bd=1)
        monitoring_placeholder.pack(fill='both', expand=True)
        
        placeholder_label = tk.Label(monitoring_placeholder, 
                                    text="📊\n\nNetwork Monitoring Interface\nReal-time packet capture and analysis\nProtocol statistics and visualization",
                                    bg=ModernTheme.DARKER_BG,
                                    fg=ModernTheme.TEXT_SECONDARY,
                                    font=('Segoe UI', 14))
        placeholder_label.pack(expand=True)
    
    def create_configuration_tab(self):
        """Create configuration management tab"""
        config_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(config_frame, text="⚙️ Configuration")
        
        content_frame = tk.Frame(config_frame, bg=ModernTheme.DARK_BG)
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header_label = ttk.Label(content_frame, text="⚙️ Configuration Management", 
                                style='Title.TLabel')
        header_label.pack(anchor='w', pady=(0, 20))
        
        # Configuration interface placeholder
        config_placeholder = tk.Frame(content_frame, bg=ModernTheme.DARKER_BG, relief='solid', bd=1)
        config_placeholder.pack(fill='both', expand=True)
        
        placeholder_label = tk.Label(config_placeholder, 
                                    text="⚙️\n\nConfiguration Management\nSave and load test configurations\nTemplate management\nProfile settings",
                                    bg=ModernTheme.DARKER_BG,
                                    fg=ModernTheme.TEXT_SECONDARY,
                                    font=('Segoe UI', 14))
        placeholder_label.pack(expand=True)
    
    def create_logs_tab(self):
        """Create logs and output tab"""
        logs_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(logs_frame, text="📝 Logs")
        
        content_frame = tk.Frame(logs_frame, bg=ModernTheme.DARK_BG)
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header with controls
        header_frame = tk.Frame(content_frame, bg=ModernTheme.DARK_BG)
        header_frame.pack(fill='x', pady=(0, 15))
        
        logs_label = ttk.Label(header_frame, text="📝 System Logs & Output", 
                              style='Title.TLabel')
        logs_label.pack(side='left')
        
        # Log controls
        controls_frame = tk.Frame(header_frame, bg=ModernTheme.DARK_BG)
        controls_frame.pack(side='right')
        
        ttk.Button(controls_frame, text="Clear Logs",
                  style='Warning.TButton',
                  command=self.clear_logs).pack(side='right', padx=(10, 0))
        
        ttk.Button(controls_frame, text="Export Logs",
                  style='Modern.TButton',
                  command=self.export_logs).pack(side='right')
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(content_frame,
                                                 bg=ModernTheme.DARKER_BG,
                                                 fg=ModernTheme.TEXT_PRIMARY,
                                                 font=('Consolas', 10),
                                                 state='disabled')
        self.log_text.pack(fill='both', expand=True)
        
        # Add initial log entries
        self.add_log("Advanced Network Protocol Testing Suite initialized", "INFO")
        self.add_log("All subsystems ready for testing", "SUCCESS")
    
    def create_status_bar(self):
        """Create enhanced status bar"""
        self.status_bar = tk.Frame(self.root, bg=ModernTheme.DARKER_BG, height=30)
        self.status_bar.pack(fill='x', side='bottom')
        self.status_bar.pack_propagate(False)
        
        # Left side - Status text
        left_status = tk.Frame(self.status_bar, bg=ModernTheme.DARKER_BG)
        left_status.pack(side='left', fill='y')
        
        self.status_var = tk.StringVar(value="Ready for network testing")
        status_label = tk.Label(left_status, textvariable=self.status_var,
                               bg=ModernTheme.DARKER_BG,
                               fg=ModernTheme.TEXT_SECONDARY,
                               font=('Segoe UI', 9))
        status_label.pack(side='left', padx=15, pady=6)
        
        # Right side - Version and info
        right_status = tk.Frame(self.status_bar, bg=ModernTheme.DARKER_BG)
        right_status.pack(side='right', fill='y')
        
        version_label = tk.Label(right_status, text="v2.0.0 Professional",
                                bg=ModernTheme.DARKER_BG,
                                fg=ModernTheme.TEXT_DISABLED,
                                font=('Segoe UI', 8))
        version_label.pack(side='right', padx=15, pady=6)
    
    # Simulation methods
    def simulate_bgp(self):
        """Simulate BGP protocol"""
        self.clear_routing_config()
        self.add_log("BGP simulation started", "INFO")
        self.create_bgp_simulation_interface()
    
    def simulate_ospf(self):
        """Simulate OSPF protocol"""
        self.clear_routing_config()
        self.add_log("OSPF simulation started", "INFO")
        self.create_ospf_simulation_interface()
    
    def simulate_eigrp(self):
        """Simulate EIGRP protocol"""
        self.clear_routing_config()
        self.add_log("EIGRP simulation started", "INFO")
        self.create_eigrp_simulation_interface()
    
    def simulate_isis(self):
        """Simulate IS-IS protocol"""
        self.clear_routing_config()
        self.add_log("IS-IS simulation started", "INFO")
        self.create_isis_simulation_interface()
    
    def simulate_rip(self):
        """Simulate RIP protocol"""
        self.clear_routing_config()
        self.add_log("RIP simulation started", "INFO")
        self.create_rip_simulation_interface()
    
    def clear_routing_config(self):
        """Clear routing configuration area"""
        for widget in self.routing_config_frame.winfo_children():
            widget.destroy()
    
    def create_bgp_simulation_interface(self):
        """Create BGP simulation interface"""
        # BGP Configuration
        config_frame = tk.Frame(self.routing_config_frame, bg=ModernTheme.DARKER_BG)
        config_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        ttk.Label(config_frame, text="🔄 BGP Configuration", 
                 style='SubHeader.TLabel').pack(anchor='w', pady=(0, 15))
        
        # AS Number
        as_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        as_frame.pack(fill='x', pady=5)
        
        ttk.Label(as_frame, text="AS Number:", style='Modern.TLabel').pack(side='left')
        as_var = tk.StringVar(value="65001")
        ttk.Entry(as_frame, textvariable=as_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Router ID
        router_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        router_frame.pack(fill='x', pady=5)
        
        ttk.Label(router_frame, text="Router ID:", style='Modern.TLabel').pack(side='left')
        router_var = tk.StringVar(value="1.1.1.1")
        ttk.Entry(router_frame, textvariable=router_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Neighbor
        neighbor_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        neighbor_frame.pack(fill='x', pady=5)
        
        ttk.Label(neighbor_frame, text="Neighbor IP:", style='Modern.TLabel').pack(side='left')
        neighbor_var = tk.StringVar(value="192.168.1.2")
        ttk.Entry(neighbor_frame, textvariable=neighbor_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Start simulation button
        start_btn = ttk.Button(config_frame, text="Start BGP Simulation",
                              style='Success.TButton',
                              command=lambda: self.start_bgp_simulation(as_var.get(), router_var.get(), neighbor_var.get()))
        start_btn.pack(pady=20)
    
    def create_ospf_simulation_interface(self):
        """Create OSPF simulation interface"""
        config_frame = tk.Frame(self.routing_config_frame, bg=ModernTheme.DARKER_BG)
        config_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        ttk.Label(config_frame, text="🕷 OSPF Configuration", 
                 style='SubHeader.TLabel').pack(anchor='w', pady=(0, 15))
        
        # Process ID
        process_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        process_frame.pack(fill='x', pady=5)
        
        ttk.Label(process_frame, text="Process ID:", style='Modern.TLabel').pack(side='left')
        process_var = tk.StringVar(value="1")
        ttk.Entry(process_frame, textvariable=process_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Area
        area_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        area_frame.pack(fill='x', pady=5)
        
        ttk.Label(area_frame, text="Area:", style='Modern.TLabel').pack(side='left')
        area_var = tk.StringVar(value="0.0.0.0")
        ttk.Entry(area_frame, textvariable=area_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Network
        network_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        network_frame.pack(fill='x', pady=5)
        
        ttk.Label(network_frame, text="Network:", style='Modern.TLabel').pack(side='left')
        network_var = tk.StringVar(value="192.168.1.0")
        ttk.Entry(network_frame, textvariable=network_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Start simulation button
        start_btn = ttk.Button(config_frame, text="Start OSPF Simulation",
                              style='Success.TButton',
                              command=lambda: self.start_ospf_simulation(process_var.get(), area_var.get(), network_var.get()))
        start_btn.pack(pady=20)
    
    def create_eigrp_simulation_interface(self):
        """Create EIGRP simulation interface"""
        config_frame = tk.Frame(self.routing_config_frame, bg=ModernTheme.DARKER_BG)
        config_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        ttk.Label(config_frame, text="⚡ EIGRP Configuration", 
                 style='SubHeader.TLabel').pack(anchor='w', pady=(0, 15))
        
        # AS Number
        as_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        as_frame.pack(fill='x', pady=5)
        
        ttk.Label(as_frame, text="AS Number:", style='Modern.TLabel').pack(side='left')
        as_var = tk.StringVar(value="100")
        ttk.Entry(as_frame, textvariable=as_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Router ID
        router_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        router_frame.pack(fill='x', pady=5)
        
        ttk.Label(router_frame, text="Router ID:", style='Modern.TLabel').pack(side='left')
        router_var = tk.StringVar(value="10.1.1.1")
        ttk.Entry(router_frame, textvariable=router_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Network
        network_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        network_frame.pack(fill='x', pady=5)
        
        ttk.Label(network_frame, text="Network:", style='Modern.TLabel').pack(side='left')
        network_var = tk.StringVar(value="192.168.1.0")
        ttk.Entry(network_frame, textvariable=network_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Wildcard Mask
        wildcard_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        wildcard_frame.pack(fill='x', pady=5)
        
        ttk.Label(wildcard_frame, text="Wildcard Mask:", style='Modern.TLabel').pack(side='left')
        wildcard_var = tk.StringVar(value="0.0.0.255")
        ttk.Entry(wildcard_frame, textvariable=wildcard_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # K Values
        k_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        k_frame.pack(fill='x', pady=5)
        
        ttk.Label(k_frame, text="K Values (K1 K2 K3 K4 K5):", style='Modern.TLabel').pack(side='left')
        k_var = tk.StringVar(value="1 0 1 0 0")
        ttk.Entry(k_frame, textvariable=k_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Hello Interval
        hello_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        hello_frame.pack(fill='x', pady=5)
        
        ttk.Label(hello_frame, text="Hello Interval:", style='Modern.TLabel').pack(side='left')
        hello_var = tk.StringVar(value="5")
        ttk.Entry(hello_frame, textvariable=hello_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Hold Time
        hold_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        hold_frame.pack(fill='x', pady=5)
        
        ttk.Label(hold_frame, text="Hold Time:", style='Modern.TLabel').pack(side='left')
        hold_var = tk.StringVar(value="15")
        ttk.Entry(hold_frame, textvariable=hold_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Start simulation button
        start_btn = ttk.Button(config_frame, text="Start EIGRP Simulation",
                              style='Success.TButton',
                              command=lambda: self.start_eigrp_simulation(
                                  as_var.get(), router_var.get(), network_var.get(), 
                                  wildcard_var.get(), k_var.get(), hello_var.get(), hold_var.get()))
        start_btn.pack(pady=20)
    
    def create_isis_simulation_interface(self):
        """Create IS-IS simulation interface"""
        config_frame = tk.Frame(self.routing_config_frame, bg=ModernTheme.DARKER_BG)
        config_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        ttk.Label(config_frame, text="🔗 IS-IS Configuration", 
                 style='SubHeader.TLabel').pack(anchor='w', pady=(0, 15))
        
        # System ID
        system_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        system_frame.pack(fill='x', pady=5)
        
        ttk.Label(system_frame, text="System ID:", style='Modern.TLabel').pack(side='left')
        system_var = tk.StringVar(value="1921.6800.1001")
        ttk.Entry(system_frame, textvariable=system_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Area ID
        area_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        area_frame.pack(fill='x', pady=5)
        
        ttk.Label(area_frame, text="Area ID:", style='Modern.TLabel').pack(side='left')
        area_var = tk.StringVar(value="49.0001")
        ttk.Entry(area_frame, textvariable=area_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Level
        level_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        level_frame.pack(fill='x', pady=5)
        
        ttk.Label(level_frame, text="IS-IS Level:", style='Modern.TLabel').pack(side='left')
        level_var = tk.StringVar(value="level-1-2")
        level_combo = ttk.Combobox(level_frame, textvariable=level_var,
                                  values=["level-1", "level-2", "level-1-2"],
                                  style='Modern.TCombobox', width=12, state='readonly')
        level_combo.pack(side='right')
        
        # NET Address
        net_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        net_frame.pack(fill='x', pady=5)
        
        ttk.Label(net_frame, text="NET Address:", style='Modern.TLabel').pack(side='left')
        net_var = tk.StringVar(value="49.0001.1921.6800.1001.00")
        ttk.Entry(net_frame, textvariable=net_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Hello Interval
        hello_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        hello_frame.pack(fill='x', pady=5)
        
        ttk.Label(hello_frame, text="Hello Interval:", style='Modern.TLabel').pack(side='left')
        hello_var = tk.StringVar(value="10")
        ttk.Entry(hello_frame, textvariable=hello_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # LSP Interval
        lsp_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        lsp_frame.pack(fill='x', pady=5)
        
        ttk.Label(lsp_frame, text="LSP Interval:", style='Modern.TLabel').pack(side='left')
        lsp_var = tk.StringVar(value="33")
        ttk.Entry(lsp_frame, textvariable=lsp_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Authentication
        auth_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        auth_frame.pack(fill='x', pady=5)
        
        ttk.Label(auth_frame, text="Authentication Key:", style='Modern.TLabel').pack(side='left')
        auth_var = tk.StringVar(value="cisco123")
        ttk.Entry(auth_frame, textvariable=auth_var, style='Modern.TEntry', width=15, show="*").pack(side='right')
        
        # Start simulation button
        start_btn = ttk.Button(config_frame, text="Start IS-IS Simulation",
                              style='Success.TButton',
                              command=lambda: self.start_isis_simulation(
                                  system_var.get(), area_var.get(), level_var.get(),
                                  net_var.get(), hello_var.get(), lsp_var.get(), auth_var.get()))
        start_btn.pack(pady=20)
    
    def create_rip_simulation_interface(self):
        """Create RIP simulation interface"""
        config_frame = tk.Frame(self.routing_config_frame, bg=ModernTheme.DARKER_BG)
        config_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        ttk.Label(config_frame, text="📍 RIP Configuration", 
                 style='SubHeader.TLabel').pack(anchor='w', pady=(0, 15))
        
        # RIP Version
        version_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        version_frame.pack(fill='x', pady=5)
        
        ttk.Label(version_frame, text="RIP Version:", style='Modern.TLabel').pack(side='left')
        version_var = tk.StringVar(value="2")
        version_combo = ttk.Combobox(version_frame, textvariable=version_var,
                                    values=["1", "2"],
                                    style='Modern.TCombobox', width=12, state='readonly')
        version_combo.pack(side='right')
        
        # Network
        network_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        network_frame.pack(fill='x', pady=5)
        
        ttk.Label(network_frame, text="Network:", style='Modern.TLabel').pack(side='left')
        network_var = tk.StringVar(value="192.168.1.0")
        ttk.Entry(network_frame, textvariable=network_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Update Timer
        update_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        update_frame.pack(fill='x', pady=5)
        
        ttk.Label(update_frame, text="Update Timer:", style='Modern.TLabel').pack(side='left')
        update_var = tk.StringVar(value="30")
        ttk.Entry(update_frame, textvariable=update_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Invalid Timer
        invalid_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        invalid_frame.pack(fill='x', pady=5)
        
        ttk.Label(invalid_frame, text="Invalid Timer:", style='Modern.TLabel').pack(side='left')
        invalid_var = tk.StringVar(value="180")
        ttk.Entry(invalid_frame, textvariable=invalid_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Holddown Timer
        holddown_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        holddown_frame.pack(fill='x', pady=5)
        
        ttk.Label(holddown_frame, text="Holddown Timer:", style='Modern.TLabel').pack(side='left')
        holddown_var = tk.StringVar(value="180")
        ttk.Entry(holddown_frame, textvariable=holddown_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Flush Timer
        flush_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        flush_frame.pack(fill='x', pady=5)
        
        ttk.Label(flush_frame, text="Flush Timer:", style='Modern.TLabel').pack(side='left')
        flush_var = tk.StringVar(value="240")
        ttk.Entry(flush_frame, textvariable=flush_var, style='Modern.TEntry', width=15).pack(side='right')
        
        # Authentication
        auth_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        auth_frame.pack(fill='x', pady=5)
        
        ttk.Label(auth_frame, text="Authentication:", style='Modern.TLabel').pack(side='left')
        auth_var = tk.StringVar(value="none")
        auth_combo = ttk.Combobox(auth_frame, textvariable=auth_var,
                                 values=["none", "text", "md5"],
                                 style='Modern.TCombobox', width=12, state='readonly')
        auth_combo.pack(side='right')
        
        # Authentication Key
        key_frame = tk.Frame(config_frame, bg=ModernTheme.DARKER_BG)
        key_frame.pack(fill='x', pady=5)
        
        ttk.Label(key_frame, text="Auth Key:", style='Modern.TLabel').pack(side='left')
        key_var = tk.StringVar(value="")
        ttk.Entry(key_frame, textvariable=key_var, style='Modern.TEntry', width=15, show="*").pack(side='right')
        
        # Start simulation button
        start_btn = ttk.Button(config_frame, text="Start RIP Simulation",
                              style='Success.TButton',
                              command=lambda: self.start_rip_simulation(
                                  version_var.get(), network_var.get(), update_var.get(),
                                  invalid_var.get(), holddown_var.get(), flush_var.get(),
                                  auth_var.get(), key_var.get()))
        start_btn.pack(pady=20)
    
    # Simulation start methods
    def start_bgp_simulation(self, as_number, router_id, neighbor_ip):
        """Start BGP simulation"""
        self.add_log(f"Starting BGP simulation: AS {as_number}, Router ID {router_id}, Neighbor {neighbor_ip}", "INFO")
        # Implement BGP simulation logic here
        
    def start_ospf_simulation(self, process_id, area, network):
        """Start OSPF simulation"""
        self.add_log(f"Starting OSPF simulation: Process {process_id}, Area {area}, Network {network}", "INFO")
        # Implement OSPF simulation logic here
        
        if SCAPY_AVAILABLE:
            try:
                # Create OSPF Hello packet
                ospf_hello = IP(dst="224.0.0.5") / OSPF_Hdr(type=1) / OSPF_Hello(
                    router=process_id,
                    area=area,
                    hello=10,
                    dead=40
                )
                
                # Send OSPF Hello
                send(ospf_hello, verbose=False)
                self.add_log(f"OSPF Hello packet sent - Area: {area}", "SUCCESS")
                
            except Exception as e:
                self.add_log(f"Error in OSPF simulation: {e}", "ERROR")
        else:
            self.add_log("Scapy not available for OSPF simulation", "WARNING")
    
    def start_eigrp_simulation(self, as_number, router_id, network, wildcard, k_values, hello_interval, hold_time):
        """Start EIGRP simulation"""
        self.add_log(f"Starting EIGRP simulation: AS {as_number}, Router ID {router_id}, Network {network}/{wildcard}", "INFO")
        
        if SCAPY_AVAILABLE:
            try:
                # Create EIGRP Hello packet
                eigrp_hello = IP(dst="224.0.0.10") / EIGRP_Hdr(
                    opcode=5,  # Hello
                    asn=int(as_number),
                    seq=0,
                    ack=0
                ) / EIGRP_Hello(
                    holdtime=int(hold_time),
                    k1=int(k_values.split()[0]) if len(k_values.split()) > 0 else 1,
                    k2=int(k_values.split()[1]) if len(k_values.split()) > 1 else 0,
                    k3=int(k_values.split()[2]) if len(k_values.split()) > 2 else 1,
                    k4=int(k_values.split()[3]) if len(k_values.split()) > 3 else 0,
                    k5=int(k_values.split()[4]) if len(k_values.split()) > 4 else 0
                )
                
                # Send EIGRP Hello
                send(eigrp_hello, verbose=False)
                self.add_log(f"EIGRP Hello packet sent - AS: {as_number}, Hold Time: {hold_time}s", "SUCCESS")
                
            except Exception as e:
                self.add_log(f"Error in EIGRP simulation: {e}", "ERROR")
        else:
            self.add_log("Scapy not available for EIGRP simulation", "WARNING")
    
    def start_isis_simulation(self, system_id, area_id, level, net_address, hello_interval, lsp_interval, auth_key):
        """Start IS-IS simulation"""
        self.add_log(f"Starting IS-IS simulation: System ID {system_id}, Area {area_id}, Level {level}", "INFO")
        
        if SCAPY_AVAILABLE:
            try:
                # Create IS-IS Hello packet (simplified)
                isis_hello = Ether(dst="01:80:c2:00:00:14") / Raw(load=f"IS-IS Hello: {system_id}")
                
                # Send IS-IS Hello
                sendp(isis_hello, verbose=False)
                self.add_log(f"IS-IS Hello packet sent - System ID: {system_id}, Level: {level}", "SUCCESS")
                
            except Exception as e:
                self.add_log(f"Error in IS-IS simulation: {e}", "ERROR")
        else:
            self.add_log("Scapy not available for IS-IS simulation", "WARNING")
    
    def start_rip_simulation(self, version, network, update_timer, invalid_timer, holddown_timer, flush_timer, auth_type, auth_key):
        """Start RIP simulation"""
        self.add_log(f"Starting RIP simulation: Version {version}, Network {network}, Update Timer {update_timer}s", "INFO")
        
        if SCAPY_AVAILABLE:
            try:
                # Create RIP packet
                rip_packet = IP(dst="224.0.0.9") / UDP(dport=520) / Raw(load=f"RIPv{version} Update")
                
                # Send RIP packet
                send(rip_packet, verbose=False)
                self.add_log(f"RIP v{version} packet sent - Network: {network}, Timer: {update_timer}s", "SUCCESS")
                
            except Exception as e:
                self.add_log(f"Error in RIP simulation: {e}", "ERROR")
        else:
            self.add_log("Scapy not available for RIP simulation", "WARNING")
    
    # Script execution methods
    def execute_cisco_script(self):
        """Execute selected Cisco script"""
        if not NETMIKO_AVAILABLE:
            messagebox.showerror("Error", "Netmiko library required for device connection")
            return
        
        self.add_log("Executing Cisco script", "INFO")
        # Implement script execution
    
    def execute_juniper_script(self):
        """Execute selected Juniper script"""
        if not NETMIKO_AVAILABLE:
            messagebox.showerror("Error", "Netmiko library required for device connection")
            return
        
        self.add_log("Executing Juniper script", "INFO")
        # Implement script execution
    
    def on_script_select(self, event, vendor, scripts):
        """Handle script selection"""
        selection = event.widget.curselection()
        if selection:
            script_name = event.widget.get(selection[0])
            script_data = scripts[script_name]
            
            # Update preview
            preview_widget = getattr(self, f"{vendor.lower()}_script_preview")
            preview_widget.delete(1.0, tk.END)
            preview_widget.insert(tk.END, script_data["script"])
            
            self.add_log(f"Selected {vendor} script: {script_name}", "INFO")
    
    # Traffic generation methods
    def start_traffic_generation(self):
        """Start traffic generation with legal compliance"""
        if not self.show_legal_warning():
            return
        
        protocol = self.quick_protocol_var.get()
        target = self.quick_target_var.get()
        rate = self.quick_rate_var.get()
        
        if not target:
            messagebox.showerror("Error", "Target address is required")
            return
        
        self.is_generating = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        self.add_log(f"Started {protocol} traffic generation to {target} at {rate} pps", "SUCCESS")
        self.status_var.set(f"Generating {protocol} traffic to {target}")
    
    def stop_traffic_generation(self):
        """Stop traffic generation"""
        self.is_generating = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        self.add_log("Traffic generation stopped", "INFO")
        self.status_var.set("Ready for network testing")
    
    def show_legal_warning(self) -> bool:
        """Show comprehensive legal warning"""
        warning_text = """
⚠️ LEGAL AUTHORIZATION REQUIRED ⚠️

This advanced network testing tool generates protocol traffic and connects to network devices.

You MUST have explicit written authorization to:
• Test target networks and devices
• Connect to routers and switches
• Generate network traffic
• Execute configuration scripts

UNAUTHORIZED TESTING IS ILLEGAL and may violate:
• Computer Fraud and Abuse Act (CFAA)
• Network security policies
• Terms of service agreements
• Local and international laws

By clicking YES, you acknowledge that you:
✓ Have proper authorization for all testing targets
✓ Will only use this tool for legitimate purposes
✓ Accept full responsibility for your actions
✓ Will comply with all applicable laws and regulations

Do you have proper written authorization for this testing?
        """
        
        return messagebox.askyesno("⚠️ Legal Authorization Required", warning_text)
    
    # Utility methods
    def add_log(self, message: str, level: str = "INFO"):
        """Add enhanced log entry with colors"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Level icons
        level_icons = {
            'INFO': 'ℹ️',
            'SUCCESS': '✅',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'DEBUG': '🐛'
        }
        
        icon = level_icons.get(level, 'ℹ️')
        
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"[{timestamp}] {icon} {level}: {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        self.add_log("Logs cleared", "INFO")
    
    def export_logs(self):
        """Export logs to file"""
        filename = filedialog.asksaveasfilename(
            title="Export Logs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self.add_log(f"Logs exported to {filename}", "SUCCESS")
            except Exception as e:
                self.add_log(f"Failed to export logs: {e}", "ERROR")
    
    def process_messages(self):
        """Process messages from worker threads"""
        try:
            while True:
                message = self.message_queue.get_nowait()
                # Process different message types
                if message['type'] == 'stats_update':
                    self.update_statistics(message)
                elif message['type'] == 'error':
                    self.add_log(f"Error: {message['message']}", "ERROR")
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_messages)
    
    def update_statistics(self, message):
        """Update statistics display"""
        self.stats['packets_sent'] = message.get('packets', 0)
        self.stats['bytes_sent'] += message.get('bytes', 0)
        
        # Update metric cards
        self.packets_card.update_value(str(self.stats['packets_sent']))
        self.bytes_card.update_value(self.format_bytes(self.stats['bytes_sent']))
    
    def format_bytes(self, bytes_count: int) -> str:
        """Format bytes in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_count < 1024:
                return f"{bytes_count:.1f}"
            bytes_count /= 1024
        return f"{bytes_count:.1f}TB"
    
    def check_dependencies(self):
        """Check for required dependencies and show status"""
        missing = []
        
        if not SCAPY_AVAILABLE:
            missing.append("scapy")
        if not NETMIKO_AVAILABLE:
            missing.append("netmiko")
        if not PARAMIKO_AVAILABLE:
            missing.append("paramiko")
        if not MATPLOTLIB_AVAILABLE:
            missing.append("matplotlib")
        if not REQUESTS_AVAILABLE:
            missing.append("requests")
        
        if missing:
            self.add_log(f"Missing optional dependencies: {', '.join(missing)}", "WARNING")
            self.add_log("Install with: pip install " + " ".join(missing), "INFO")
        else:
            self.add_log("All dependencies available", "SUCCESS")


def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Set window icon (if available)
    try:
        root.iconbitmap('network_icon.ico')
    except:
        pass
    
    app = AdvancedNetworkTestingGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application closed")


if __name__ == "__main__":
    main()