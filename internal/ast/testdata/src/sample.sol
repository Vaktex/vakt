pragma solidity ^0.8.0;

contract Vault {
    mapping(address => uint) public balances;

    function withdraw(uint amount) public {
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok);
        balances[msg.sender] -= amount;
    }

    modifier onlyOwner() {
        _;
    }
}
