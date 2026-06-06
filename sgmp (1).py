"""
SGMP: Secure Group Messaging Protocol command-line simulation.

This prototype follows the Team23 Phase I proposal:
- RSA-PSS signatures for authentication
- X25519 ephemeral Diffie-Hellman for pairwise secrets
- HKDF-SHA256 for GroupSeed derivation
- HMAC-SHA256 sender keys
- AES-GCM encryption
- Nonce, timestamp, and sequence-number replay protection
- Rekeying on join, leave, compromise, or message threshold
"""

from __future__ import annotations

# Standard Python libraries used for encoding, JSON packets, randomness, and timing
import base64
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

# Cryptography library imports used for signatures, key exchange, encryption, and key derivation
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, hmac, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


# Configuration values used by the protocol
FRESHNESS_WINDOW_SECONDS = 30
MESSAGE_ROTATION_THRESHOLD = 5


# Convert bytes to Base64 so they can be stored inside JSON-like packets
def b64(data: bytes) -> str:
    """Encode bytes so they can be placed inside JSON-like message packets."""
    return base64.b64encode(data).decode("ascii")


# Convert Base64 text back into raw bytes
def unb64(data: str) -> bytes:
    """Decode base64 text back into raw cryptographic bytes."""
    return base64.b64decode(data.encode("ascii"))


# Create a stable JSON format before signing or verifying data
def canonical_json(data: dict) -> bytes:
    """Serialize data in a stable order before signing or verifying it."""
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


# Convert an RSA public key into bytes if it needs to be exported
def rsa_public_bytes(public_key) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


# Convert an X25519 public key into raw bytes
def x25519_public_bytes(public_key: x25519.X25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


# Load an X25519 public key from raw bytes
def load_x25519_public(raw: bytes) -> x25519.X25519PublicKey:
    return x25519.X25519PublicKey.from_public_bytes(raw)


# Create an RSA-PSS signature over a protocol payload
def sign(private_key, payload: bytes) -> bytes:
    """Create an RSA-PSS signature over a protocol payload."""
    return private_key.sign(
        payload,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )


# Verify that a received RSA-PSS signature is valid
def verify(public_key, signature: bytes, payload: bytes) -> bool:
    """Verify an RSA-PSS signature and return True only when it is valid."""
    try:
        public_key.verify(
            signature,
            payload,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return True
    except InvalidSignature:
        return False


# Derive the shared GroupSeed for the current epoch and current membership
def hkdf_group_seed(secret_material: bytes, membership: Iterable[str], epoch: int) -> bytes:
    """Derive the epoch GroupSeed from DH material and current membership."""
    member_context = ",".join(sorted(membership)).encode("utf-8")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"SGMP-v1-group-seed" + epoch.to_bytes(8, "big"),
        info=b"members:" + member_context,
    ).derive(secret_material)


# Derive a unique AES-GCM sender key for each participant
def derive_sender_key(group_seed: bytes, sender_id: str) -> bytes:
    """Derive one AES-GCM sender key for a specific group member."""
    mac = hmac.HMAC(group_seed, hashes.SHA256())
    mac.update(b"SGMP-v1-sender-key:" + sender_id.encode("utf-8"))
    return mac.finalize()


# Stores the public identity information that each participant shares with the group
@dataclass
class PublicIdentity:
    """Public information broadcast by a participant during identity exchange."""

    user_id: str
    rsa_public_key: object
    dh_public_key: x25519.X25519PublicKey
    signature: bytes


# Represents one group member and its local cryptographic state
@dataclass
class Participant:
    """Represents one SGMP group member and its local cryptographic state."""

    user_id: str
    rsa_private_key: object = field(init=False)
    rsa_public_key: object = field(init=False)
    dh_private_key: x25519.X25519PrivateKey = field(init=False)
    dh_public_key: x25519.X25519PublicKey = field(init=False)
    sender_keys: Dict[str, bytes] = field(default_factory=dict)
    used_nonces: Dict[str, set] = field(default_factory=dict)
    last_sequence: Dict[str, int] = field(default_factory=dict)
    outbound_sequence: int = 0
    accepted_messages_in_epoch: int = 0

    def __post_init__(self) -> None:
        # Generate RSA keys for digital signatures
        self.rsa_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.rsa_public_key = self.rsa_private_key.public_key()

        # Generate the first X25519 key pair for Diffie-Hellman
        self.rotate_dh_key()

    def rotate_dh_key(self) -> None:
        # Create a fresh X25519 key pair for the current epoch
        self.dh_private_key = x25519.X25519PrivateKey.generate()
        self.dh_public_key = self.dh_private_key.public_key()

    def public_identity(self) -> PublicIdentity:
        # Create a signed identity that binds the user ID with the DH public key
        dh_raw = x25519_public_bytes(self.dh_public_key)
        payload = canonical_json({"id": self.user_id, "dh_public": b64(dh_raw)})
        return PublicIdentity(
            user_id=self.user_id,
            rsa_public_key=self.rsa_public_key,
            dh_public_key=self.dh_public_key,
            signature=sign(self.rsa_private_key, payload),
        )

    def reset_epoch_state(self) -> None:
        # Clear old keys and replay protection data after rekeying
        self.sender_keys.clear()
        self.used_nonces.clear()
        self.last_sequence.clear()
        self.outbound_sequence = 0
        self.accepted_messages_in_epoch = 0

    def install_sender_keys(self, group_seed: bytes, members: Iterable[str]) -> None:
        # Install sender keys derived from the current GroupSeed
        self.sender_keys = {
            member_id: derive_sender_key(group_seed, member_id) for member_id in sorted(members)
        }

    def encrypt_message(self, plaintext: str) -> dict:
        # Make sure this sender has a valid sender key before encryption
        if self.user_id not in self.sender_keys:
            raise RuntimeError(f"{self.user_id} has no sender key for this epoch")

        # Increase the sender sequence number for message ordering
        self.outbound_sequence += 1

        # Generate a fresh random nonce for AES-GCM
        nonce = os.urandom(12)

        # Add a timestamp to help reject old delayed messages
        timestamp = int(time.time())

        # This metadata is authenticated by AES-GCM but not encrypted
        aad = canonical_json(
            {"sender": self.user_id, "timestamp": timestamp, "sequence": self.outbound_sequence}
        )

        # Encrypt the plaintext using the sender's own sender key
        ciphertext = AESGCM(self.sender_keys[self.user_id]).encrypt(
            nonce,
            plaintext.encode("utf-8"),
            aad,
        )

        # Prepare the transmitted packet
        transcript = {
            "ciphertext": b64(ciphertext),
            "nonce": b64(nonce),
            "timestamp": timestamp,
            "sequence": self.outbound_sequence,
            "sender": self.user_id,
        }

        # Sign the encrypted packet to prove the sender identity
        transcript["signature"] = b64(sign(self.rsa_private_key, canonical_json(transcript)))
        return transcript

    def decrypt_message(self, packet: dict, sender_public_key) -> str:
        # Read the claimed sender from the packet
        sender = packet["sender"]

        # Remove the signature field before verifying the signed content
        unsigned = {key: value for key, value in packet.items() if key != "signature"}

        # Verify the sender signature before doing any decryption
        if not verify(sender_public_key, unb64(packet["signature"]), canonical_json(unsigned)):
            raise ValueError("message rejected: invalid RSA signature")

        # Check that the message timestamp is still inside the freshness window
        if abs(int(time.time()) - int(packet["timestamp"])) > FRESHNESS_WINDOW_SECONDS:
            raise ValueError("message rejected: timestamp is outside freshness window")

        # Detect replayed messages using stored nonces per sender
        nonce = packet["nonce"]
        self.used_nonces.setdefault(sender, set())
        if nonce in self.used_nonces[sender]:
            raise ValueError("message rejected: replayed nonce")
        self.used_nonces[sender].add(nonce)

        # Make sure messages from the same sender arrive in the expected order
        expected = self.last_sequence.get(sender, 0) + 1
        if int(packet["sequence"]) != expected:
            raise ValueError(
                f"message rejected: sequence {packet['sequence']} received, expected {expected}"
            )
        self.last_sequence[sender] = int(packet["sequence"])

        # Check that the receiver has a sender key for this sender
        if sender not in self.sender_keys:
            raise ValueError("message rejected: unknown sender key")

        # Decrypt the message only after all verification checks pass
        aad = canonical_json(
            {"sender": sender, "timestamp": packet["timestamp"], "sequence": packet["sequence"]}
        )
        plaintext = AESGCM(self.sender_keys[sender]).decrypt(
            unb64(packet["nonce"]),
            unb64(packet["ciphertext"]),
            aad,
        )

        # Count accepted messages in this epoch
        self.accepted_messages_in_epoch += 1
        return plaintext.decode("utf-8")


# Simulates the group environment and manages membership changes
class SGMPGroup:
    """Manages the simulated group, membership changes, and key rotations."""

    def __init__(self, members: List[str]):
        # The project scope requires a small group of 4 to 8 members
        if not 4 <= len(members) <= 8:
            raise ValueError("proposal scope requires a small group of 4 to 8 members")

        # Start from epoch zero, then rotate keys for the initial setup
        self.epoch = 0

        # Create a Participant object for each member name
        self.participants = {member: Participant(member) for member in members}

        # This directory stores verified public identities
        self.public_directory: Dict[str, PublicIdentity] = {}

        # Perform the first key rotation
        self.rotate_keys("initial setup")

    def authenticate_identities(self) -> None:
        # Verify all public identities before starting communication
        directory = {}
        for participant in self.participants.values():
            identity = participant.public_identity()

            # Rebuild the signed identity payload
            payload = canonical_json(
                {
                    "id": identity.user_id,
                    "dh_public": b64(x25519_public_bytes(identity.dh_public_key)),
                }
            )

            # Reject the identity if the RSA signature is invalid
            if not verify(identity.rsa_public_key, identity.signature, payload):
                raise RuntimeError(f"identity verification failed for {identity.user_id}")

            # Store the verified identity
            directory[identity.user_id] = identity

        self.public_directory = directory

    def rotate_keys(self, reason: str) -> None:
        # Rekey the group whenever membership or epoch changes
        self.epoch += 1

        # Every active participant gets a fresh DH key and clears old epoch data
        for participant in self.participants.values():
            participant.rotate_dh_key()
            participant.reset_epoch_state()

        # Authenticate the new public identities
        self.authenticate_identities()

        # Derive a new GroupSeed for the current members
        members = sorted(self.participants)
        group_seed = self._derive_epoch_group_seed(members)

        # Install the new sender keys for each participant
        for user_id, participant in self.participants.items():
            participant.install_sender_keys(group_seed, members)

        print(f"[epoch {self.epoch}] key rotation complete: {reason}")

    def _derive_epoch_group_seed(self, members: List[str]) -> bytes:
       
        # Combine all pairwise DH secrets into one deterministic transcript
        pairwise = []
        for left_index, left_id in enumerate(members):
            for right_id in members[left_index + 1 :]:
                left = self.participants[left_id]
                right_public = self.public_directory[right_id].dh_public_key

                # Compute the X25519 shared secret for this pair
                secret = left.dh_private_key.exchange(right_public)

                # Add member IDs with the secret to keep the order clear and consistent
                pairwise.append(left_id.encode("utf-8") + b"|" + right_id.encode("utf-8") + b"|" + secret)

        # Use HKDF to derive the final GroupSeed from the pairwise material
        return hkdf_group_seed(b"".join(pairwise), members, self.epoch)

    def broadcast(self, sender_id: str, plaintext: str) -> dict:
        # Simulate sending one encrypted message to all active members
        sender = self.participants[sender_id]
        packet = sender.encrypt_message(plaintext)
        print(f"\n{sender_id} broadcasts encrypted message: {packet['ciphertext'][:48]}...")

        # Each receiver verifies and decrypts the packet
        for receiver_id, receiver in self.participants.items():
            if receiver_id == sender_id:
                continue
            try:
                message = receiver.decrypt_message(
                    packet,
                    self.public_directory[sender_id].rsa_public_key,
                )
                print(f"  {receiver_id} accepted from {sender_id}: {message}")
            except Exception as exc:
                print(f"  {receiver_id} rejected from {sender_id}: {exc}")

        # Rotate keys when the message threshold is reached
        if sender.outbound_sequence % MESSAGE_ROTATION_THRESHOLD == 0:
            self.rotate_keys(f"message threshold reached by {sender_id}")

        return packet

    def join(self, user_id: str) -> None:
        # Add a new member and trigger rekeying for backward secrecy
        if user_id in self.participants:
            raise ValueError(f"{user_id} is already a member")
        if len(self.participants) >= 8:
            raise ValueError("proposal scope allows at most 8 members")

        self.participants[user_id] = Participant(user_id)
        self.rotate_keys(f"{user_id} joined")

    def leave(self, user_id: str) -> None:
        # Remove a member and generate fresh keys for forward secrecy
        if user_id not in self.participants:
            raise ValueError(f"{user_id} is not a member")

        del self.participants[user_id]

        # Rekey the remaining group members
        if self.participants:
            self.rotate_keys(f"{user_id} left")

    def compromise(self, user_id: str) -> None:
        # Treat a compromised device as unsafe and remove it from the active group
        self.leave(user_id)
        print(f"{user_id} must rejoin with a fresh RSA key, DH key, and identity.")


# Demonstrates the main SGMP protocol features
def run_demo() -> None:
    """Run a small scenario that demonstrates the main protocol requirements."""

    # Create the first group with four members
    group = SGMPGroup(["Layan", "Ghala", "Hour", "Atheer"])

    # Layan sends the first encrypted message
    first_packet = group.broadcast("Layan", "Meeting notes are ready.")

    # Reusing the same packet should fail because the nonce was already used
    print("\nReplay attack test:")
    try:
        group.participants["Ghala"].decrypt_message(
            first_packet,
            group.public_directory["Layan"].rsa_public_key,
        )
    except Exception as exc:
        print(f"  replay correctly rejected: {exc}")

    # Normal encrypted group message
    group.broadcast("Ghala", "I reviewed the protocol design.")

    # A new member joins, so the group rotates keys
    group.join("Muzun")

    # The new member can only use the new epoch keys
    group.broadcast("Muzun", "I joined after the rekey and cannot read old packets.")

    # A member leaves, so the group rotates keys again
    group.leave("Hour")

    # Future messages are protected from the removed member
    group.broadcast("Atheer", "Future messages use a new epoch without Hour.")

    # A compromised member is removed and emergency rekeying is performed
    group.compromise("Ghala")

    # Message after emergency rekeying
    group.broadcast("Layan", "Emergency rekey is complete.")


if __name__ == "__main__":
    run_demo()
