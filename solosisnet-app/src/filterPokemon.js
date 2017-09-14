import pokemonList from './pokemon.json';

export default function filterPokemon(searchText) {
  return pokemonList.filter((pokemon) => pokemon.Name.includes(searchText))
}