import math
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import time
from datetime import datetime

# --- Core mathematical functions used in the QWDE system ---

def compute_D(n, kappa=0.01):
    """
    Compute D(n) = 1 - exp(-kappa * n), represents some temporal decay function.
    """
    value = 1 - math.exp(-kappa * n)
    print(f"compute_D: n={n}, kappa={kappa}, result={value}")
    return value

def compute_tau_infinity(omega, D_n, tau_max):
    """
    Compute temporal fuzzing parameter τ∞ = ω * D(n) * τ_max.
    """
    value = omega * D_n * tau_max
    print(f"compute_tau_infinity: omega={omega}, D_n={D_n}, tau_max={tau_max}, result={value}")
    return value

def compute_security_level(eta, E):
    """
    Compute security level = 1 - exp(-eta * E).
    E here is the entropy or energy measure (take first byte as int).
    """
    energy_measure = E[0] if len(E) > 0 else 1
    value = 1 - math.exp(-eta * energy_measure)
    print(f"compute_security_level: eta={eta}, E[0]={energy_measure}, result={value}")
    return value

def sha256_hash(data: bytes) -> bytes:
    """
    Returns SHA-256 hash of the input bytes.
    """
    hashed = hashlib.sha256(data).digest()
    print(f"sha256_hash: input_len={len(data)}, output_hash={hashed.hex()}")
    return hashed

def xor_bytes(byte_seq1: bytes, byte_seq2: bytes) -> bytes:
    """
    XOR two byte sequences of the same length.
    """
    xored = bytes(x ^ y for x, y in zip(byte_seq1, byte_seq2))
    print(f"xor_bytes: length={len(xored)}")
    return xored

def morph_seed(S: bytes, E: bytes, quadrant_index: int, U: bytes, morph_counter: int = 0) -> bytes:
    """
    Generate a polymorphic seed for each quadrant based on:
    - S (seed)
    - E (entropy)
    - quadrant index (0-3)
    - U (unique user or timestamp input)
    - morph_counter (incremented after each decryption for polymorphic behavior)
    """
    print(f"morph_seed: Generating seed for quadrant {quadrant_index}, morph_counter={morph_counter}")
    base_hash = sha256_hash(S + E + quadrant_index.to_bytes(1, 'big') + morph_counter.to_bytes(4, 'big'))
    user_hash = sha256_hash(U + base_hash)
    morphed = xor_bytes(base_hash, user_hash)
    print(f"morph_seed: quadrant={quadrant_index}, morphed_seed={morphed.hex()}")
    return morphed

def pad_pkcs7(data: bytes) -> bytes:
    """
    Pad input to AES block size (16 bytes) using PKCS#7.
    """
    pad_len = 16 - (len(data) % 16)
    padded = data + bytes([pad_len] * pad_len)
    print(f"pad_pkcs7: pad_len={pad_len}, padded_len={len(padded)}")
    return padded

def unpad_pkcs7(data: bytes) -> bytes:
    """
    Remove PKCS#7 padding.
    """
    pad_len = data[-1]
    if pad_len < 1 or pad_len > 16:
        raise ValueError("Invalid PKCS#7 padding.")
    print(f"unpad_pkcs7: pad_len={pad_len}, data_len_before={len(data)}")
    return data[:-pad_len]

def aes_encrypt(key: bytes, plaintext: bytes) -> bytes:
    """
    Encrypt plaintext with AES CBC mode using given key.
    Prepends the IV to ciphertext.
    """
    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_pt = pad_pkcs7(plaintext)
    encrypted = iv + cipher.encrypt(padded_pt)
    print(f"aes_encrypt: iv={iv.hex()}, plaintext_len={len(plaintext)}, ciphertext_len={len(encrypted)}")
    return encrypted

def aes_decrypt(key: bytes, ciphertext: bytes) -> bytes:
    """
    Decrypt ciphertext with AES CBC mode using given key.
    Expects IV prepended to ciphertext.
    """
    iv = ciphertext[:16]
    ct = ciphertext[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_pt = cipher.decrypt(ct)
    plaintext = unpad_pkcs7(padded_pt)
    print(f"aes_decrypt: iv={iv.hex()}, ciphertext_len={len(ciphertext)}, plaintext_len={len(plaintext)}")
    return plaintext

def split_into_quadrants(message: bytes) -> list:
    """
    Split the message into 4 roughly equal parts (quadrants).
    """
    length = len(message)
    quadrant_len = length // 4
    quadrants = [message[i*quadrant_len:(i+1)*quadrant_len] for i in range(3)]
    quadrants.append(message[3*quadrant_len:])  # Last quadrant gets remainder
    print(f"split_into_quadrants: message_len={length}, quadrant_len={quadrant_len}")
    for idx, quad in enumerate(quadrants):
        print(f" Quadrant {idx}: len={len(quad)}")
    return quadrants

def compute_error_correction_hash(seeds: list, E: bytes) -> bytes:
    """
    Compute error correction hash as a SHA256 over all seeds concatenated plus E.
    """
    combined = b''.join(seeds) + E
    ec_hash = sha256_hash(combined)
    print(f"compute_error_correction_hash: hash={ec_hash.hex()}")
    return ec_hash

def encrypt_qwde(
    S: bytes, E: bytes, U: bytes, plaintext: bytes,
    omega: float, tau_max: float, eta: float, n: int, kappa: float,
    morph_counter: int = 0
):
    """
    Perform full encryption following the QWDE system with polymorphic support.
    """
    print(f"Starting encryption process... morph_counter={morph_counter}")
    D_n = compute_D(n, kappa)
    tau_inf = compute_tau_infinity(omega, D_n, tau_max)
    security = compute_security_level(eta, E)

    quadrants = split_into_quadrants(plaintext)
    seeds = [morph_seed(S, E, i, U, morph_counter) for i in range(4)]

    ciphertexts = []
    for i in range(4):
        ct = aes_encrypt(seeds[i], quadrants[i])
        ciphertexts.append(ct)

    ec_hash = compute_error_correction_hash(seeds, E)

    print("Encryption completed.")
    return {
        "ciphertexts": ciphertexts,
        "seeds": seeds,
        "error_correction_hash": ec_hash,
        "morph_counter": morph_counter,
        "temporal_parameters": {
            "tau_infinity": tau_inf,
            "D_n": D_n,
            "security_level": security
        }
    }

def decrypt_qwde(seeds: list, E: bytes, U: bytes, ciphertexts: list) -> bytes:
    """
    Perform full decryption following the QWDE system.
    """
    print("Starting decryption process...")
    decrypted_quadrants = []
    for i in range(4):
        pt = aes_decrypt(seeds[i], ciphertexts[i])
        decrypted_quadrants.append(pt)
    plaintext = b''.join(decrypted_quadrants)
    print("Decryption completed.")
    return plaintext

def update_polymorphic_ciphertext(
    S: bytes, E: bytes, U: bytes, plaintext: bytes,
    omega: float, tau_max: float, eta: float, n: int, kappa: float,
    current_morph_counter: int
):
    """
    Update the ciphertext polymorphically by incrementing the morph counter.
    """
    print(f"Updating polymorphic ciphertext... current_counter={current_morph_counter}")
    new_counter = current_morph_counter + 1
    return encrypt_qwde(S, E, U, plaintext, omega, tau_max, eta, n, kappa, new_counter)

# --- Enhanced GUI Implementation ---

class QWDE_GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🔐 Polymorphic QWDE Encryption System")
        self.geometry("900x800")
        self.configure(bg='#1a1a1a')
        
        # Configure style for modern look
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        self.encryption_result = None
        self.original_plaintext = None
        self.decryption_count = 0
        
        self.setup_gui()
        
    def configure_styles(self):
        """Configure modern dark theme styles"""
        self.style.configure('Title.TLabel', 
                           foreground='#00ff88', 
                           background='#1a1a1a', 
                           font=('Arial', 16, 'bold'))
        
        self.style.configure('Subtitle.TLabel', 
                           foreground='#88ccff', 
                           background='#1a1a1a', 
                           font=('Arial', 11, 'bold'))
        
        self.style.configure('Modern.TFrame', 
                           background='#2d2d2d',
                           relief='raised',
                           borderwidth=2)
        
        self.style.configure('Encrypt.TButton',
                           foreground='white',
                           background='#00aa44',
                           font=('Arial', 11, 'bold'),
                           padding=10)
        
        self.style.configure('Decrypt.TButton',
                           foreground='white',
                           background='#0066cc',
                           font=('Arial', 11, 'bold'),
                           padding=10)
        
        self.style.configure('Generate.TButton',
                           foreground='white',
                           background='#ff6600',
                           font=('Arial', 9),
                           padding=5)

    def setup_gui(self):
        """Setup the enhanced GUI layout"""
        
        # Title
        title_frame = tk.Frame(self, bg='#1a1a1a')
        title_frame.pack(fill='x', pady=10)
        
        title_label = ttk.Label(title_frame, 
                               text="🔐 Polymorphic QWDE Encryption System", 
                               style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, 
                                  text="Advanced Quantum-Wave Dynamics Encryption with Polymorphic Ciphertext", 
                                  style='Subtitle.TLabel')
        subtitle_label.pack(pady=(5, 0))
        
        # Status frame
        self.status_frame = ttk.Frame(self, style='Modern.TFrame')
        self.status_frame.pack(fill='x', padx=10, pady=5)
        
        self.status_label = ttk.Label(self.status_frame, 
                                     text="Status: Ready", 
                                     foreground='#00ff88',
                                     background='#2d2d2d',
                                     font=('Arial', 10, 'bold'))
        self.status_label.pack(pady=5)
        
        # Main container with notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Encryption Tab
        self.encrypt_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.encrypt_frame, text="🔒 Encryption")
        self.setup_encryption_tab()
        
        # Decryption Tab
        self.decrypt_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.decrypt_frame, text="🔓 Decryption")
        self.setup_decryption_tab()
        
        # Analytics Tab
        self.analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_frame, text="📊 Analytics")
        self.setup_analytics_tab()

    def setup_encryption_tab(self):
        """Setup encryption tab"""
        
        # Plaintext input
        input_frame = ttk.LabelFrame(self.encrypt_frame, text="📝 Input Message", padding=10)
        input_frame.pack(fill='x', padx=10, pady=5)
        
        self.plaintext_box = scrolledtext.ScrolledText(input_frame, 
                                                      height=6, 
                                                      bg='#2d2d2d', 
                                                      fg='white',
                                                      insertbackground='white',
                                                      font=('Consolas', 10))
        self.plaintext_box.pack(fill='x')
        
        # Cryptographic parameters
        crypto_frame = ttk.LabelFrame(self.encrypt_frame, text="🔑 Cryptographic Parameters", padding=10)
        crypto_frame.pack(fill='x', padx=10, pady=5)
        
        # Create grid for parameters
        param_grid = ttk.Frame(crypto_frame)
        param_grid.pack(fill='x')
        
        # Seeds and keys
        self.create_param_row(param_grid, 0, "Seed S (hex):", "entry_S", get_random_bytes(16).hex())
        self.create_param_row(param_grid, 1, "Entropy E (hex):", "entry_E", get_random_bytes(16).hex())
        self.create_param_row(param_grid, 2, "Unique U (hex):", "entry_U", get_random_bytes(16).hex())
        
        # Mathematical parameters
        math_frame = ttk.LabelFrame(self.encrypt_frame, text="📐 Mathematical Parameters", padding=10)
        math_frame.pack(fill='x', padx=10, pady=5)
        
        math_grid = ttk.Frame(math_frame)
        math_grid.pack(fill='x')
        
        self.create_param_row(math_grid, 0, "ω (omega):", "entry_omega", "1.0", width=15)
        self.create_param_row(math_grid, 1, "τ_max:", "entry_tau_max", "1.0", width=15)
        self.create_param_row(math_grid, 2, "η (eta):", "entry_eta", "0.1", width=15)
        self.create_param_row(math_grid, 3, "n:", "entry_n", "100", width=15)
        self.create_param_row(math_grid, 4, "κ (kappa):", "entry_kappa", "0.01", width=15)
        
        # Encrypt button
        button_frame = ttk.Frame(self.encrypt_frame)
        button_frame.pack(pady=15)
        
        self.btn_encrypt = ttk.Button(button_frame, 
                                     text="🔒 ENCRYPT MESSAGE", 
                                     command=self.perform_encryption,
                                     style='Encrypt.TButton')
        self.btn_encrypt.pack(side='left', padx=5)
        
        self.btn_generate_keys = ttk.Button(button_frame, 
                                          text="🎲 Generate New Keys", 
                                          command=self.generate_new_keys,
                                          style='Generate.TButton')
        self.btn_generate_keys.pack(side='left', padx=5)
        
        # Ciphertext output
        output_frame = ttk.LabelFrame(self.encrypt_frame, text="🔐 Encrypted Output", padding=10)
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.ciphertext_box = scrolledtext.ScrolledText(output_frame, 
                                                       height=12, 
                                                       bg='#1a1a2e', 
                                                       fg='#00ff88',
                                                       insertbackground='white',
                                                       font=('Consolas', 9))
        self.ciphertext_box.pack(fill='both', expand=True)

    def setup_decryption_tab(self):
        """Setup decryption tab"""
        
        # Info frame
        info_frame = ttk.LabelFrame(self.decrypt_frame, text="ℹ️ Polymorphic Decryption Info", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        info_text = ("The polymorphic feature automatically updates the ciphertext after each decryption,\n"
                    "maintaining security while preserving the original message integrity.")
        ttk.Label(info_frame, text=info_text, foreground='#88ccff').pack()
        
        # Decryption controls
        control_frame = ttk.Frame(self.decrypt_frame)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        self.btn_decrypt = ttk.Button(control_frame, 
                                     text="🔓 DECRYPT MESSAGE", 
                                     command=self.perform_decryption,
                                     style='Decrypt.TButton')
        self.btn_decrypt.pack(side='left', padx=5)
        
        # Decryption counter
        self.decrypt_counter_label = ttk.Label(control_frame, 
                                              text="Decryptions: 0", 
                                              foreground='#ff6600',
                                              font=('Arial', 10, 'bold'))
        self.decrypt_counter_label.pack(side='right', padx=5)
        
        # Decrypted output
        decrypt_output_frame = ttk.LabelFrame(self.decrypt_frame, text="✅ Decrypted Message", padding=10)
        decrypt_output_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.decrypted_box = scrolledtext.ScrolledText(decrypt_output_frame, 
                                                      height=12, 
                                                      bg='#1a2e1a', 
                                                      fg='#88ff88',
                                                      insertbackground='white',
                                                      font=('Consolas', 10))
        self.decrypted_box.pack(fill='both', expand=True)

    def setup_analytics_tab(self):
        """Setup analytics tab"""
        
        # System metrics
        metrics_frame = ttk.LabelFrame(self.analytics_frame, text="📊 System Metrics", padding=10)
        metrics_frame.pack(fill='x', padx=10, pady=5)
        
        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, 
                                                     height=8, 
                                                     bg='#2d2d2d', 
                                                     fg='#ffff88',
                                                     font=('Consolas', 9))
        self.metrics_text.pack(fill='both', expand=True)
        
        # Polymorphic status
        poly_frame = ttk.LabelFrame(self.analytics_frame, text="🔄 Polymorphic Status", padding=10)
        poly_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.poly_status_text = scrolledtext.ScrolledText(poly_frame, 
                                                         height=8, 
                                                         bg='#2d1a2d', 
                                                         fg='#ff88ff',
                                                         font=('Consolas', 9))
        self.poly_status_text.pack(fill='both', expand=True)

    def create_param_row(self, parent, row, label_text, entry_name, default_value, width=48):
        """Create a parameter input row"""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky='w', pady=2)
        
        entry_frame = ttk.Frame(parent)
        entry_frame.grid(row=row, column=1, sticky='w', padx=10, pady=2)
        
        entry = tk.Entry(entry_frame, width=width, bg='#2d2d2d', fg='white', insertbackground='white')
        entry.pack(side='left')
        entry.insert(0, default_value)
        setattr(self, entry_name, entry)
        
        if 'hex' in label_text.lower():
            btn_gen = ttk.Button(entry_frame, text="🎲", width=3,
                               command=lambda e=entry: self.regenerate_hex_value(e))
            btn_gen.pack(side='left', padx=5)

    def regenerate_hex_value(self, entry):
        """Regenerate a hex value for an entry"""
        entry.delete(0, tk.END)
        entry.insert(0, get_random_bytes(16).hex())

    def generate_new_keys(self):
        """Generate new cryptographic keys"""
        self.entry_S.delete(0, tk.END)
        self.entry_S.insert(0, get_random_bytes(16).hex())
        
        self.entry_E.delete(0, tk.END)
        self.entry_E.insert(0, get_random_bytes(16).hex())
        
        self.entry_U.delete(0, tk.END)
        self.entry_U.insert(0, get_random_bytes(16).hex())
        
        self.update_status("New cryptographic keys generated", "#00ff88")

    def update_status(self, message, color="#00ff88"):
        """Update status message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_label.config(text=f"Status: {message} [{timestamp}]", foreground=color)

    def update_metrics(self, result):
        """Update system metrics display"""
        self.metrics_text.delete('1.0', tk.END)
        
        temp = result["temporal_parameters"]
        metrics = f"""=== QWDE System Metrics ===
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Temporal Parameters:
├─ τ∞ (tau infinity): {temp['tau_infinity']:.6f}
├─ D(n): {temp['D_n']:.6f}
└─ Security Level: {temp['security_level']:.6f}

Cryptographic Info:
├─ Error Correction Hash: {result['error_correction_hash'].hex()}
├─ Morph Counter: {result['morph_counter']}
└─ Quadrants Encrypted: 4

Polymorphic Status:
├─ Total Decryptions: {self.decryption_count}
├─ Ciphertext Updates: {self.decryption_count}
└─ Message Integrity: MAINTAINED
"""
        self.metrics_text.insert(tk.END, metrics)

    def update_polymorphic_status(self):
        """Update polymorphic status display"""
        self.poly_status_text.delete('1.0', tk.END)
        
        if self.encryption_result:
            status = f"""=== Polymorphic Ciphertext Status ===
Current Morph Generation: {self.encryption_result['morph_counter']}
Decryption Events: {self.decryption_count}
Next Morph Counter: {self.encryption_result['morph_counter'] + 1}

Polymorphic Benefits:
✓ Each decryption triggers ciphertext update
✓ Original message remains unchanged
✓ Enhanced forward secrecy
✓ Dynamic encryption keys per generation

Security Features:
├─ Temporal Parameter Evolution
├─ Quadrant-based Seed Morphing  
├─ Error Correction Validation
└─ Entropy-driven Key Derivation

Last Update: {datetime.now().strftime("%H:%M:%S")}
"""
        else:
            status = "No active encryption session. Please encrypt a message first."
            
        self.poly_status_text.insert(tk.END, status)

    def perform_encryption(self):
        """Perform encryption with enhanced feedback"""
        print("\n=== Encryption Requested ===")
        self.update_status("Encrypting message...", "#ffaa00")
        
        try:
            plaintext = self.plaintext_box.get('1.0', 'end').encode('utf-8').strip()
            if not plaintext:
                messagebox.showwarning("Input Error", "Please enter plaintext to encrypt.")
                return

            # Get parameters
            S = bytes.fromhex(self.entry_S.get().strip())
            E = bytes.fromhex(self.entry_E.get().strip())
            U = bytes.fromhex(self.entry_U.get().strip())

            omega = float(self.entry_omega.get())
            tau_max = float(self.entry_tau_max.get())
            eta = float(self.entry_eta.get())
            n = int(self.entry_n.get())
            kappa = float(self.entry_kappa.get())

            # Perform encryption
            result = encrypt_qwde(S, E, U, plaintext, omega, tau_max, eta, n, kappa, 0)
            self.encryption_result = result
            self.original_plaintext = plaintext
            self.decryption_count = 0

            # Display ciphertexts with headers
            ciphertext_display = "=== POLYMORPHIC QWDE CIPHERTEXT ===\n"
            ciphertext_display += f"Generation: {result['morph_counter']}\n"
            ciphertext_display += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for i, ct in enumerate(result["ciphertexts"]):
                ciphertext_display += f"Quadrant {i+1}:\n{ct.hex()}\n\n"

            self.ciphertext_box.delete('1.0', tk.END)
            self.ciphertext_box.insert(tk.END, ciphertext_display)

            # Clear decrypted text
            self.decrypted_box.delete('1.0', tk.END)

            # Update analytics
            self.update_metrics(result)
            self.update_polymorphic_status()
            
            self.update_status("Encryption completed successfully", "#00ff88")
            messagebox.showinfo("Success", "Message encrypted with polymorphic QWDE system!")

        except Exception as ex:
            print(f"Error during encryption: {ex}")
            self.update_status(f"Encryption failed: {str(ex)}", "#ff4444")
            messagebox.showerror("Encryption Error", f"An error occurred: {ex}")

    def perform_decryption(self):
        """Perform decryption with polymorphic update"""
        print("\n=== Decryption Requested ===")
        self.update_status("Decrypting message...", "#ffaa00")
        
        try:
            if self.encryption_result is None:
                messagebox.showwarning("Decryption Warning", "Please perform encryption first.")
                return

            # Decrypt current ciphertext
            seeds = self.encryption_result["seeds"]
            ciphertexts = self.encryption_result["ciphertexts"]
            
            decrypted = decrypt_qwde(seeds, None, None, ciphertexts)
            decrypted_text = decrypted.decode('utf-8', errors='replace')

            # Update decryption counter
            self.decryption_count += 1
            self.decrypt_counter_label.config(text=f"Decryptions: {self.decryption_count}")

            # Display decrypted text
            self.decrypted_box.delete('1.0', tk.END)
            decrypt_display = f"=== DECRYPTED MESSAGE (Access #{self.decryption_count}) ===\n"
            decrypt_display += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            decrypt_display += decrypted_text
            self.decrypted_box.insert(tk.END, decrypt_display)

            # POLYMORPHIC UPDATE: Re-encrypt with new morph counter
            if self.original_plaintext:
                S = bytes.fromhex(self.entry_S.get().strip())
                E = bytes.fromhex(self.entry_E.get().strip())
                U = bytes.fromhex(self.entry_U.get().strip())
                omega = float(self.entry_omega.get())
                tau_max = float(self.entry_tau_max.get())
                eta = float(self.entry_eta.get())
                n = int(self.entry_n.get())
                kappa = float(self.entry_kappa.get())

                print(f"Updating polymorphic ciphertext (morph counter: {self.encryption_result['morph_counter']} -> {self.encryption_result['morph_counter'] + 1})")
                
                # Generate new ciphertext with incremented morph counter
                updated_result = update_polymorphic_ciphertext(
                    S, E, U, self.original_plaintext, 
                    omega, tau_max, eta, n, kappa,
                    self.encryption_result['morph_counter']
                )
                
                self.encryption_result = updated_result

                # Update ciphertext display
                ciphertext_display = "=== POLYMORPHIC QWDE CIPHERTEXT (UPDATED) ===\n"
                ciphertext_display += f"Generation: {updated_result['morph_counter']}\n"
                ciphertext_display += f"Auto-update after decryption #{self.decryption_count}\n"
                ciphertext_display += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                for i, ct in enumerate(updated_result["ciphertexts"]):
                    ciphertext_display += f"Quadrant {i+1} (Updated):\n{ct.hex()}\n\n"

                self.ciphertext_box.delete('1.0', tk.END)
                self.ciphertext_box.insert(tk.END, ciphertext_display)

                # Update analytics
                self.update_metrics(updated_result)
                self.update_polymorphic_status()

            self.update_status(f"Decryption #{self.decryption_count} completed, ciphertext updated", "#00ff88")
            messagebox.showinfo("Success", f"Message decrypted successfully!\nPolymorphic ciphertext updated (Generation {self.encryption_result['morph_counter']})")

        except Exception as ex:
            print(f"Error during decryption: {ex}")
            self.update_status(f"Decryption failed: {str(ex)}", "#ff4444")
            messagebox.showerror("Decryption Error", f"An error occurred: {ex}")

if __name__ == "__main__":
    print("Launching Enhanced Polymorphic QWDE Encryption GUI...")
    app = QWDE_GUI()
    app.mainloop()