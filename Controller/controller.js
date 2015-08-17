//
// controller.js
// Firefly [v1]
//
// Copyright (c) 2015 Mihir Garimella.
//

var stdin = process.stdin;
var stdout = process.stdout;
var chalk = require('chalk');

var net = require('net');
var dns = require('dns');
var client = new net.Socket();
var connected = false;
var logging = false;

// Find the ODROID's IP address.
dns.lookup('odroid.local', function(error, address, family) {
	if (error) {
		console.log(chalk.red.bold('Couldn\'t find ODROID.'));
	} else {
		// Connect to the ODROID via TCP.
		client.connect(3000, address, function() {
			console.log(chalk.green.bold('Connected to ODROID at ' + address + '.'));
			stdin.resume();
			showPrompt();
			connected = true;
		});
		client.on('data', function(data) {
			console.log('\b\b' + data);
			showPrompt();
		});
		client.on('error', function() {
			console.log(chalk.red.bold(connected ? '\b\bDisconnected from ODROID.' : 'Couldn\'t connect to ODROID.'));
		});
		client.on('close', function() {
			process.exit(0);
		});
	}
});

function showPrompt() {
	// Show a prompt, and wait for input on stdin.
	stdout.write("> ");
	stdin.once('data', function(data) {
		var input = data.toString().replace(/\n$/, '');
		if (input == 'exit') {
			client.destroy();
		} else {
			if (input == 'flat' || input == 'f') {
				console.log(chalk.green('Flat trim.'));
			} else if (input == 'reset' || input == 'r') {
				console.log(chalk.green('Resetting.'));
			} else if (input == 'takeoff' || input == 't') {
				console.log(chalk.green('Taking off.'));
			} else if (input == 'move' || input == 'm') {
				console.log(chalk.green('Moving.'));
			} else if (input == 'land' || input == 'l') {
				console.log(chalk.green('Landing.'));
			} else if (input == 'stop' || input == 's') {
				console.log(chalk.green('Stopping.'));
			} else {
				console.log(chalk.red('Didn\'t recognize command ' + input + '.'));
			}
			client.write(input.charAt(0));
			showPrompt();
		}
	});
}