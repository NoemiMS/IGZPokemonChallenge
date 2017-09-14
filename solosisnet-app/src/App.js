import React, { Component } from 'react';
import './App.css';
import SearchInput from './searchInput';
import filterPokemon from './filterPokemon'

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      filteredPokemon: filterPokemon(''),
    };
  }

  handleSearchChange = (event) => {
    this.setState({
      filteredPokemon: filterPokemon(event.target.value),
    });
    console.log(filterPokemon(event.target.value))
  }
  render() {
    return (
      <div className="App">
        <div className="App-header">
          <img src="./assets/img/solosis-logo.png" className="App-logo" alt="logo" />
          <h2 className="pokemonFontBold">SolosisNet</h2>
        </div>
        <div className="App-intro">
          <h3>IGZ Pokemon Challenge</h3>
          <p>SolosisNet is a ML project developed by Ana de Prado, ML gym leader , and Noe Medina, junior trainner. </p>
        </div>
        <div className="combats-wrapper">
          <SearchInput
            textChange={this.handleSearchChange}
          />

        </div>
      </div>
    );
  }
}

export default App;
